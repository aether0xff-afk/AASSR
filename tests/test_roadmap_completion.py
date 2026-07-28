from __future__ import annotations

from dataclasses import replace

from aassr_v2.ablations import (
    imagination_ablation_matrix,
    representation_ablation_matrix,
)
from aassr_v2.action_plugins import (
    ActionRegistry,
    ActionSchema,
    ParameterSpec,
    PluginOutcome,
    SlotCandidateResolver,
    parameter_lessons,
)
from aassr_v2.adapters import (
    AuthorizedAssessmentPlugin,
    DryRunTransport,
    MinecraftControlPlugin,
)
from aassr_v2.counterexamples import (
    LearnableVsRandomWorld,
    LongDependencyWorld,
    opaque_name_map,
    permuted_positions,
)
from aassr_v2.curriculum_engine import (
    AcademyStage,
    CurriculumTeacher,
)
from aassr_v2.feature_memory import OnlineFeatureMemory
from aassr_v2.goals import (
    GoalGenerator,
    GoalKind,
    GoalSet,
    GoalStateScorer,
)
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.knowledge import KnowledgeStore
from aassr_v2.learning import AdvancedTransitionEvaluator
from aassr_v2.sandbox import SandboxEnv
from aassr_v2.skills import (
    SKILL_VERB,
    SkillAwareProphecy,
    SkillLibrary,
)
from aassr_v2.types import (
    Action,
    Prediction,
    StateSnapshot,
    TransitionTrace,
)


class DummyPlugin:
    plugin_id = "dummy"

    def __init__(self) -> None:
        self.schema = ActionSchema(
            "dummy",
            "operate",
            (
                ParameterSpec(
                    "subject",
                    "subject",
                    "identifier",
                ),
                ParameterSpec(
                    "mode",
                    "option",
                    "identifier",
                    required=False,
                    default="safe",
                ),
            ),
        )
        self.state = StateSnapshot((0.0,))

    def schemas(self):
        return (self.schema,)

    def enumerate_values(self, state, schema, parameter):
        return ()

    def execute(self, action):
        return PluginOutcome(self.state)


def test_plugin_schema_is_syntax_only_and_action_keeps_parameters() -> None:
    plugin = DummyPlugin()
    registry = ActionRegistry()
    registry.register(plugin)
    action = plugin.schema.build({"subject": "value-17"})
    assert action.verb_name == "operate"
    assert action.parameters == {
        "subject": "value-17",
        "mode": "safe",
    }
    assert registry.execute(action).snapshot == plugin.state
    lessons = parameter_lessons(plugin.schema)
    assert len(lessons) == 2
    assert "유용한지는 알려주지 않는다" in lessons[0].instruction


def test_feature_memory_learns_action_slot_role() -> None:
    memory = OnlineFeatureMemory(cluster_threshold=0.5)
    memory.observe_information(
        "credential-a",
        ("secret", "authentication"),
    )
    memory.observe_information(
        "credential-b",
        ("secret", "authentication"),
    )
    memory.observe_information(
        "host-a",
        ("network", "destination"),
    )
    for _ in range(4):
        memory.observe_use(
            "credential-a",
            action_id="connect",
            slot="authentication",
            value=1.0,
        )
    memory.observe_use(
        "host-a",
        action_id="connect",
        slot="authentication",
        value=-1.0,
    )
    ranked = memory.rank_for_slot(
        "connect",
        "authentication",
        ("host-a", "credential-b", "credential-a"),
        limit=2,
    )
    assert ranked[0] == "credential-a"
    schema = ActionSchema(
        "x",
        "connect",
        (ParameterSpec("auth", "authentication"),),
    )
    assignments = SlotCandidateResolver(
        memory,
        per_slot_limit=2,
    ).resolve(
        schema,
        {
            "credential-a": "A",
            "credential-b": "B",
            "host-a": "H",
        },
    )
    assert assignments[0]["auth"] == "A"


def test_goal_gap_scores_desired_state_not_action_name() -> None:
    before = StateSnapshot(
        (0.0,),
        frozenset(),
        (Action("wait"),),
        0.0,
    )
    desired = StateSnapshot(
        (1.0,),
        frozenset({"inventory:item-7"}),
        (
            Action(
                "use",
                parameters={"item": "item-7"},
            ),
        ),
        1.0,
    )
    generated = GoalGenerator.from_desired_state(
        before,
        desired,
    )
    assert {goal.kind for goal in generated} >= {
        GoalKind.FACT_PRESENT,
        GoalKind.ACTION_AVAILABLE,
        GoalKind.GOAL_PROGRESS,
    }
    goals = GoalSet(
        {goal.goal_id: goal for goal in generated}
    )
    assert (
        GoalStateScorer(goals).score(
            before,
            Action("anything"),
            desired,
        )
        > 10
    )


def make_trace(
    index: int,
    before: StateSnapshot,
    action: Action,
    after: StateSnapshot,
    goal_id: str = "g",
) -> TransitionTrace:
    return TransitionTrace(
        f"t{index}",
        before,
        action,
        (Prediction(after),),
        after,
        after.facts - before.facts,
        before.facts - after.facts,
        (),
        False,
        goal_ids=(goal_id,),
    )


class ExactProphecy:
    name = "exact"

    def __init__(self, transitions) -> None:
        self.transitions = transitions

    def initial_memory(self):
        return ()

    def predict_step(
        self,
        state,
        action,
        *,
        memory,
        samples,
    ):
        from aassr_v2.prophecy import ProphecyStep

        next_state = self.transitions.get(
            (state.vector, action.signature),
            state,
        )
        return ProphecyStep(
            (Prediction(next_state, 1.0, "exact"),),
            memory + (action.signature,),
        )

    def predict(self, state, action, *, samples):
        return self.predict_step(
            state,
            action,
            memory=(),
            samples=samples,
        ).predictions

    def learn(self, *args):
        pass


def test_repeated_goal_sequence_becomes_imagined_skill() -> None:
    state0 = StateSnapshot(
        (0.0,),
        frozenset({"ready"}),
        (Action("a"),),
        0.0,
    )
    state1 = StateSnapshot(
        (0.5,),
        frozenset({"ready", "mid"}),
        (Action("b"),),
        0.5,
    )
    state2 = StateSnapshot(
        (1.0,),
        frozenset({"ready", "mid", "done"}),
        (),
        1.0,
    )
    traces = (
        make_trace(1, state0, Action("a"), state1),
        make_trace(2, state1, Action("b"), state2),
    )
    library = SkillLibrary(promotion_successes=2)
    assert (
        library.observe_goal_completion(
            traces,
            achieved_goal_ids=("g",),
        )
        is None
    )
    skill = library.observe_goal_completion(
        traces,
        achieved_goal_ids=("g",),
    )
    assert skill is not None
    augmented = library.augment_state(state0)
    skill_action = next(
        action
        for action in augmented.available_actions
        if action.verb_name == SKILL_VERB
    )
    prophecy = SkillAwareProphecy(
        ExactProphecy(
            {
                (
                    state0.vector,
                    Action("a").signature,
                ): state1,
                (
                    state1.vector,
                    Action("b").signature,
                ): state2,
            }
        ),
        library,
    )
    prediction = prophecy.predict(
        augmented,
        skill_action,
        samples=1,
    )[0]
    assert prediction.next_state.goal_progress == 1.0


def test_online_gru_learns_real_transition_and_branch_memory() -> None:
    model = OnlineGRUProphecy(
        2,
        hidden_size=6,
        action_feature_size=4,
        learning_rate=0.03,
        seed=2,
    )
    state = StateSnapshot(
        (0.0, 0.0),
        available_actions=(Action("advance"),),
    )
    next_state = StateSnapshot(
        (1.0, 0.0),
        facts=frozenset({"advanced"}),
        available_actions=(Action("advance"),),
        goal_progress=0.5,
    )
    initial_error = sum(
        (left - right) ** 2
        for left, right in zip(
            model.predict_vector(
                state,
                Action("advance"),
            ),
            next_state.vector,
            strict=True,
        )
    )
    for _ in range(80):
        model.reset_sequence()
        model.learn(
            state,
            Action("advance"),
            next_state,
        )
    final_error = sum(
        (left - right) ** 2
        for left, right in zip(
            model.predict_vector(
                state,
                Action("advance"),
            ),
            next_state.vector,
            strict=True,
        )
    )
    assert final_error < initial_error
    step = model.predict_step(
        state,
        Action("advance"),
        memory=model.initial_memory(),
        samples=1,
    )
    assert step.memory.hidden != model.initial_memory().hidden
    assert step.predictions[0].next_state == next_state


def test_sandbox_exposes_generic_combine_not_recipe() -> None:
    environment = SandboxEnv()
    schemas = {
        schema.action_id: schema
        for schema in environment.plugin.schemas()
    }
    assert set(schemas) == {
        "observe",
        "break",
        "place",
        "combine",
    }
    assert "recipe" not in {
        parameter.name
        for parameter in schemas["combine"].parameters
    }
    environment.step(
        schemas["break"].build(
            {"subject": "resource_a"}
        )
    )
    environment.step(
        schemas["break"].build(
            {"subject": "resource_b"}
        )
    )
    outcome = environment.step(
        schemas["combine"].build(
            {"items": ("resource_a", "resource_b")}
        )
    )
    assert not outcome.error
    assert environment.snapshot().goal_progress == 1.0


def test_curriculum_teaches_syntax_without_optimal_values() -> None:
    registry = ActionRegistry()
    registry.register(DummyPlugin())
    teacher = CurriculumTeacher.from_registry(registry)
    assert (
        teacher.lessons[-1].stage
        is AcademyStage.PLUGIN_PARAMETERS
    )
    assert all(
        "value-17" not in lesson.instruction
        for lesson in teacher.lessons[-1].parameter_lessons
    )
    for _ in range(
        (len(teacher.lessons) - 1) * teacher.window
    ):
        teacher.observe(True)
    assert (
        teacher.current.stage
        is AcademyStage.PLUGIN_PARAMETERS
    )


def test_counterexamples_cover_randomness_and_long_dependency() -> None:
    world = LearnableVsRandomWorld(seed=1)
    stable = [
        world.step(Action("probe_stable")).snapshot.vector[0]
        for _ in range(2)
    ]
    random_values = [
        world.step(Action("probe_random")).snapshot.vector[1]
        for _ in range(5)
    ]
    assert stable == [1.0, 2.0]
    assert len(set(random_values)) > 1
    chain = LongDependencyWorld(4)
    for stage in range(4):
        chain.step(
            Action(
                "inspect",
                parameters={"stage": stage},
            )
        )
        chain.step(
            Action(
                "advance",
                parameters={"stage": stage},
            )
        )
    assert chain.snapshot().goal_progress == 1.0


def test_external_plugins_are_dry_run_and_allowlisted() -> None:
    transport = DryRunTransport()
    minecraft = MinecraftControlPlugin(transport)
    look = next(
        schema
        for schema in minecraft.schemas()
        if schema.action_id == "look"
    ).build(
        {
            "yaw_delta": 1,
            "pitch_delta": -1,
        }
    )
    assert not minecraft.execute(look).error
    assessment = AuthorizedAssessmentPlugin(
        transport,
        allowlisted_targets=("lab.local",),
    )
    scan = next(
        schema
        for schema in assessment.schemas()
        if schema.action_id == "scan"
    )
    assert (
        assessment.execute(
            scan.build({"target": "outside.local"})
        ).error_code
        == "target_not_allowlisted"
    )
    assert not assessment.execute(
        scan.build({"target": "lab.local"})
    ).error


def test_ablation_matrices_cover_all_requested_components() -> None:
    imagination = imagination_ablation_matrix()
    representation = representation_ablation_matrix()
    assert any(
        item.branching_factor == 3
        and item.maximum_depth == 3
        for item in imagination
    )
    assert {item.name for item in representation} >= {
        "S0_direct",
        "S5_contextual",
        "E3_hybrid",
    }


def test_opaque_names_and_new_placements_hide_semantics() -> None:
    mapping = opaque_name_map(
        ("red_key", "blue_door"),
        seed=3,
    )
    assert set(mapping.values()) == {
        "object-0000",
        "object-0001",
    }
    assert mapping == opaque_name_map(
        ("red_key", "blue_door"),
        seed=3,
    )
    assert permuted_positions(
        3,
        width=3,
        height=2,
        seed=2,
    ) == permuted_positions(
        3,
        width=3,
        height=2,
        seed=2,
    )


class ContextProphecy:
    name = "context"

    def __init__(self) -> None:
        self.learned = False

    def predict(self, state, action, *, samples):
        target = StateSnapshot(
            (1.0,),
            facts=frozenset({"known"}),
            goal_progress=1.0,
        )
        predicted = target if self.learned else state
        return (
            Prediction(
                predicted,
                1.0,
                (
                    "context:learned"
                    if self.learned
                    else "context:prior"
                ),
            ),
        )

    def predict_with_context(
        self,
        state,
        action,
        *,
        knowledge,
        samples,
    ):
        if knowledge.get("known") is not None:
            target = StateSnapshot(
                (1.0,),
                facts=frozenset({"known"}),
                goal_progress=1.0,
            )
            return (
                Prediction(
                    target,
                    1.0,
                    "context:knowledge",
                ),
            )
        return self.predict(
            state,
            action,
            samples=samples,
        )

    def learn(self, state, action, actual_next_state):
        self.learned = True


class OneStepEnv:
    def __init__(self) -> None:
        self.done = False

    def snapshot(self):
        if self.done:
            return StateSnapshot(
                (1.0,),
                facts=frozenset({"known"}),
                goal_progress=1.0,
            )
        return StateSnapshot(
            (0.0,),
            available_actions=(Action("discover"),),
        )

    def step(self, action):
        from types import SimpleNamespace

        self.done = True
        return SimpleNamespace(
            snapshot=self.snapshot(),
            added_facts=frozenset({"known"}),
            removed_facts=frozenset(),
            unlocked_actions=(),
            error=False,
            reward=1.0,
        )


def test_evaluator_separates_knowledge_model_and_logs_jsonl(
    tmp_path,
) -> None:
    from aassr_v2.serialization import JsonlLedgerWriter

    prophecy = ContextProphecy()
    environment = OneStepEnv()
    knowledge = KnowledgeStore()
    evaluator = AdvancedTransitionEvaluator(
        prophecy,
        logger=JsonlLedgerWriter(
            tmp_path / "trace.jsonl"
        ),
        samples=1,
    )
    result = evaluator.execute(
        environment,
        Action("discover"),
        knowledge,
    )
    assert result.effect.knowledge_only_gain > 0
    assert (
        result.effect.latest_prediction_after
        >= result.effect.knowledge_context_score
    )
    assert (
        tmp_path
        / "trace.jsonl"
    ).read_text(encoding="utf-8").count("transition") == 1
