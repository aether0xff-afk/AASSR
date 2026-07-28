from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
import math
import time
from typing import Any

from .actions import ActionCandidate
from .imagination import ImaginationCycle
from .knowledge import KK, KnowledgeStore
from .novelty import NoveltyMemory
from .plugins import TargetPlugin, get_plugin
from .policy import PolicyABC, PolicyView
from .prophecy import TableProphecyModel
from .reward import RewardBreakdown, RewardConfig, RewardObserver
from .tools import SafetyError, ToolExecutor, ToolResult


@dataclass
class StepRecord:
    step: int
    action: str
    template: str
    what: str
    how: str
    where: str
    status: int
    new_kv: int
    reward: float
    flag_found: bool
    solved_delta: int = 0
    solved_total: int = 0
    imagination_score: float = 1.0
    imagination_support: int = 0
    predicted_reward: float = 0.0
    predicted_solved_rate: float = 0.0
    novelty_score: float = 0.0
    novelty_bonus: float = 0.0
    novelty_signature: str = ""
    policy_reward: float = 0.0
    blocked: bool = False
    unavailable: bool = False
    syntax_penalty: float = 0.0
    candidate_count: int = 0
    evaluated_candidate_count: int = 0
    score_duration_s: float = 0.0
    tool_duration_s: float = 0.0
    policy_sampled: bool = False
    sampled_policy: str = ""
    reward_total: float = 0.0
    reward_challenge_solved: float = 0.0
    reward_challenge_progress: float = 0.0
    reward_semantic_novelty: float = 0.0
    reward_useful_observation: float = 0.0
    penalty_repeated_action: float = 0.0
    penalty_repeated_response: float = 0.0
    penalty_invalid_action: float = 0.0
    penalty_no_progress: float = 0.0
    challenge_progress: float = 0.0
    semantic_novelty: int = 0
    repeated_action: bool = False
    repeated_response: bool = False
    raw_response_hash: str = ""
    normalized_response_hash: str = ""
    predicted_immediate_solve_probability: float = 0.0
    predicted_progress_probability: float = 0.0
    predicted_eventual_solve_probability: float = 0.0
    predicted_expected_progress: float = 0.0


@dataclass
class RunResult:
    success: bool
    steps: int
    flag: str | None
    solved_challenges: list[str]
    records: list[StepRecord]
    knowledge_rows: list[dict[str, str | int | float]]


class APASSRToolDMP:
    def __init__(
        self,
        *,
        base_url: str,
        plugin: TargetPlugin | str | None = None,
        executor: ToolExecutor | None = None,
        policy: PolicyABC | None = None,
        reward_observer: RewardObserver | None = None,
        prophecy_model: TableProphecyModel | None = None,
        experience_memory: TableProphecyModel | None = None,
        imagination: ImaginationCycle | None = None,
        novelty_memory: NoveltyMemory | None = None,
        novelty_reward: float = 0.0,
        novelty_score_weight: float = 0.0,
        knowledge_reward_cap: int = 5,
        knowledge_reward_scale: float = 1.0,
        solved_reward: float = 50.0,
        step_limit: int = 30,
        candidate_eval_limit: int = 25000,
        policy_sampling_attempts: int = 512,
        reward_config: RewardConfig | None = None,
        observer: Any | None = None,
    ) -> None:
        self.plugin = get_plugin(plugin) if isinstance(plugin, str) else plugin or get_plugin("web")
        self.store = self.plugin.seed(base_url)
        self.executor = executor or ToolExecutor()
        self.policy = policy or PolicyABC()
        self.reward_observer = reward_observer
        self.prophecy_model = prophecy_model or experience_memory or TableProphecyModel()
        self.experience_memory = self.prophecy_model
        self.imagination = imagination or ImaginationCycle(self.prophecy_model)
        self.novelty_memory = novelty_memory or NoveltyMemory()
        self.novelty_reward = novelty_reward
        self.novelty_score_weight = novelty_score_weight
        self.knowledge_reward_cap = knowledge_reward_cap
        self.knowledge_reward_scale = knowledge_reward_scale
        self.solved_reward = solved_reward
        self.reward_config = reward_config or replace(RewardConfig(), challenge_solved=solved_reward)
        self.step_limit = step_limit
        self.candidate_eval_limit = candidate_eval_limit
        self.policy_sampling_attempts = policy_sampling_attempts
        self.observer = observer
        self.tried_counts: dict[str, int] = {}
        self.template_counts: dict[str, int] = {}
        self.what_counts: dict[str, int] = {}
        self.endpoint_counts: dict[str, int] = {}
        self.unavailable_tools: set[str] = set()
        self.solved_challenges: list[str] = []
        self.records: list[StepRecord] = []
        self.last_candidate_count = 0
        self.last_evaluated_candidate_count = 0
        self.last_score_duration_s = 0.0
        self.last_sampled_policy: PolicyView | None = None
        self.failure_counts: dict[str, int] = {}
        self._replay_start = len(self.prophecy_model.replay)
        self._trajectory_finalized = False
        if self.reward_observer is not None:
            self.reward_observer.reset()

    def run(self) -> RunResult:
        flag: str | None = None
        for step in range(self.step_limit):
            candidate = self.choose_candidate()
            if candidate is None:
                break
            record, result = self.execute_candidate(step, candidate)
            self.records.append(record)
            flags = self.store.values(KK.FLAG)
            if flags:
                flag = flags[0]
                return self._result(True, step + 1, flag)
        return self._result(bool(self.solved_challenges), len(self.records), flag)

    def choose_candidate(self) -> ActionCandidate | None:
        started_at = time.perf_counter()
        candidates = self._candidate_pool_for_selection()
        self.last_candidate_count = len(candidates)
        if not candidates:
            self.last_evaluated_candidate_count = 0
            self.last_score_duration_s = time.perf_counter() - started_at
            return None
        scored_candidates = self._candidate_eval_pool(candidates)
        self.last_evaluated_candidate_count = len(scored_candidates)
        scored = [(candidate, self._candidate_score_details(candidate)) for candidate in scored_candidates]
        scored.sort(key=lambda row: (-float(row[1]["final_score"]), row[0].label))
        self.last_score_duration_s = time.perf_counter() - started_at
        self._notify(
            "on_candidates_scored",
            scored=scored,
            store=self.store,
            policy=self.policy,
            prophecy=self.prophecy_model,
            dmp=self,
        )
        selected = scored[0][0]
        if self.last_sampled_policy is not None:
            self.last_sampled_policy = selected.policy
        return selected

    def _candidate_pool_for_selection(self) -> list[ActionCandidate]:
        self.last_sampled_policy = None
        sampled_candidates: dict[str, ActionCandidate] = {}
        for _ in range(max(1, self.policy_sampling_attempts)):
            policy_view = self.policy.sample_view()
            candidates = self._plugin_candidates_for_policy(policy_view)
            if candidates:
                self.last_sampled_policy = policy_view
                for candidate in candidates:
                    key = candidate.tried_key or candidate.label
                    sampled_candidates.setdefault(key, candidate)
        if sampled_candidates:
            return list(sampled_candidates.values())
        return [
            candidate
            for candidate in self.plugin.candidates(self.store)
            if candidate.tool_call.tool.value not in self.unavailable_tools
        ]

    def _plugin_candidates_for_policy(self, policy_view: PolicyView) -> list[ActionCandidate]:
        generator = getattr(self.plugin, "candidates_for_policy", None)
        if not callable(generator):
            return []
        return [
            candidate
            for candidate in generator(self.store, policy_view)
            if candidate.tool_call.tool.value not in self.unavailable_tools
        ]

    def _candidate_eval_pool(self, candidates: list[ActionCandidate]) -> list[ActionCandidate]:
        limit = self.candidate_eval_limit
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        return heapq.nlargest(limit, candidates, key=self._candidate_prefilter_score)

    def _candidate_prefilter_score(self, candidate: ActionCandidate) -> float:
        base = self.policy.score(
            candidate.policy,
            tried_count=self.tried_counts.get(candidate.tried_key, 0),
        )
        template_count = self.template_counts.get(candidate.template.value, 0)
        what_count = self.what_counts.get(candidate.policy.what.value, 0)
        endpoint = candidate.bindings.get(KK.ENDPOINT)
        endpoint_count = self.endpoint_counts.get(endpoint, 0) if endpoint else 0
        breadth = 1.0 / math.sqrt(1.0 + template_count)
        axis_breadth = 1.0 / math.sqrt(1.0 + 0.25 * what_count)
        endpoint_breadth = 1.0 / math.sqrt(1.0 + endpoint_count)
        syntax_multiplier = max(0.02, 1.0 - candidate_syntax_penalty(candidate))
        return base * breadth * axis_breadth * endpoint_breadth * syntax_multiplier

    def _candidate_score(self, candidate: ActionCandidate) -> float:
        return float(self._candidate_score_details(candidate)["final_score"])

    def _candidate_score_details(self, candidate: ActionCandidate) -> dict[str, float | int]:
        base = self.policy.score(
            candidate.policy,
            tried_count=self.tried_counts.get(candidate.tried_key, 0),
        )
        template_count = self.template_counts.get(candidate.template.value, 0)
        what_count = self.what_counts.get(candidate.policy.what.value, 0)
        endpoint = candidate.bindings.get(KK.ENDPOINT)
        endpoint_count = self.endpoint_counts.get(endpoint, 0) if endpoint else 0
        breadth = 1.0 / math.sqrt(1.0 + template_count)
        axis_breadth = 1.0 / math.sqrt(1.0 + 0.25 * what_count)
        endpoint_breadth = 1.0 / math.sqrt(1.0 + endpoint_count)
        imagination_score, prediction = self.imagination.score_multiplier(candidate)
        novelty_prediction = self.novelty_memory.predict(candidate)
        novelty_multiplier = 1.0 + self.novelty_score_weight * novelty_prediction.score
        syntax_penalty = candidate_syntax_penalty(candidate)
        syntax_multiplier = max(0.02, 1.0 - syntax_penalty)
        final_score = base * breadth * axis_breadth * endpoint_breadth * imagination_score * novelty_multiplier * syntax_multiplier
        return {
            "final_score": final_score,
            "policy_score": base,
            "breadth": breadth,
            "axis_breadth": axis_breadth,
            "endpoint_breadth": endpoint_breadth,
            "imagination_score": imagination_score,
            "imagination_support": prediction.support,
            "predicted_reward": prediction.expected_reward,
            "predicted_knowledge": prediction.expected_knowledge,
            "predicted_solved_rate": prediction.solved_rate,
            "predicted_immediate_solve_probability": prediction.immediate_solve_probability,
            "predicted_progress_probability": prediction.progress_probability,
            "predicted_eventual_solve_probability": prediction.eventual_solve_probability,
            "predicted_expected_progress": prediction.expected_progress,
            "predicted_error_rate": prediction.error_rate,
            "novelty_score": novelty_prediction.score,
            "novelty_multiplier": novelty_multiplier,
            "syntax_penalty": syntax_penalty,
            "syntax_multiplier": syntax_multiplier,
            "tried_count": self.tried_counts.get(candidate.tried_key, 0),
        }

    def execute_candidate(self, step: int, candidate: ActionCandidate) -> tuple[StepRecord, ToolResult | None]:
        self.tried_counts[candidate.tried_key] = self.tried_counts.get(candidate.tried_key, 0) + 1
        imagination_score, prediction = self.imagination.score_multiplier(candidate)
        novelty_prediction = self.novelty_memory.predict(candidate)
        self.template_counts[candidate.template.value] = self.template_counts.get(candidate.template.value, 0) + 1
        self.what_counts[candidate.policy.what.value] = self.what_counts.get(candidate.policy.what.value, 0) + 1
        endpoint = candidate.bindings.get(KK.ENDPOINT)
        if endpoint:
            self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
        syntax_penalty = candidate_syntax_penalty(candidate)
        try:
            result = self.executor.execute(candidate.tool_call)
        except SafetyError as exc:
            novelty_update = self.novelty_memory.update(
                candidate,
                status=0,
                new_kv=0,
                solved_delta=0,
            )
            breakdown = RewardBreakdown(
                penalty_invalid_action=self.reward_config.invalid_action,
                penalty_no_progress=self.reward_config.no_progress,
            )
            reward = breakdown.total
            policy_reward = reward
            record = StepRecord(
                step=step,
                action=candidate.label,
                template=candidate.template.value,
                what=candidate.policy.what.value,
                how=candidate.policy.how.value,
                where=candidate.policy.where.value,
                status=0,
                new_kv=0,
                reward=reward,
                flag_found=False,
                imagination_score=imagination_score,
                imagination_support=prediction.support,
                predicted_reward=prediction.expected_reward,
                predicted_solved_rate=prediction.solved_rate,
                novelty_score=novelty_prediction.score,
                novelty_bonus=0.0,
                novelty_signature=novelty_update.signature,
                policy_reward=policy_reward,
                blocked=True,
                syntax_penalty=syntax_penalty,
                candidate_count=self.last_candidate_count,
                evaluated_candidate_count=self.last_evaluated_candidate_count,
                score_duration_s=self.last_score_duration_s,
                policy_sampled=self.last_sampled_policy is not None,
                sampled_policy=_policy_label(self.last_sampled_policy),
                reward_total=reward,
                penalty_invalid_action=breakdown.penalty_invalid_action,
                penalty_no_progress=breakdown.penalty_no_progress,
                repeated_action=novelty_update.repeated_action,
                repeated_response=novelty_update.repeated_response,
                predicted_immediate_solve_probability=prediction.immediate_solve_probability,
                predicted_progress_probability=prediction.progress_probability,
                predicted_eventual_solve_probability=prediction.eventual_solve_probability,
                predicted_expected_progress=prediction.expected_progress,
            )
            self.policy.update(candidate.policy, policy_reward)
            self.prophecy_model.update(
                candidate,
                reward=record.reward,
                new_kv=0,
                solved_delta=0,
                status=0,
                progress=0.0,
            )
            self._notify("on_step", record=record, result=None, store=self.store, policy=self.policy, prophecy=self.prophecy_model, dmp=self)
            return record, None
        parsed = self.plugin.parse(result)
        new_kv = self.store.add_many(parsed, source=candidate.label)
        new_kv += self.store.derive()
        flag_found = bool(self.store.values(KK.FLAG))
        solved_delta = 0
        solved_total = len(self.solved_challenges)
        if self.reward_observer is not None:
            signal = self.reward_observer.observe()
            for key in signal.new_solved:
                if key not in self.solved_challenges:
                    self.solved_challenges.append(key)
            solved_delta = len(signal.new_solved)
            solved_total = signal.solved_total
        had_flag_evidence = any(fact.startswith(f"{KK.FLAG.value}:") for fact in self.novelty_memory.semantic_facts)
        new_flag_solve = flag_found and not had_flag_evidence
        learning_solved_delta = solved_delta if solved_delta > 0 else int(new_flag_solve)
        challenge_keys = tuple(signal.new_solved) if self.reward_observer is not None else ()
        if new_flag_solve:
            challenge_keys += ("flag-evidence",)
        novelty_update = self.novelty_memory.update(
            candidate,
            status=result.status,
            new_kv=new_kv,
            solved_delta=learning_solved_delta,
            response_body=result.stdout,
            semantic_items=parsed,
            challenge_keys=challenge_keys,
        )
        challenge_progress = self._challenge_progress(
            parsed=parsed, flag_found=flag_found, solved_delta=learning_solved_delta,
            meaningful_transition=novelty_update.meaningful_transition,
            semantic_novelty=novelty_update.semantic_novelty,
        )
        breakdown = self._reward_breakdown(
            candidate=candidate, result=result, solved_delta=learning_solved_delta,
            challenge_progress=challenge_progress, novelty_update=novelty_update,
            syntax_penalty=syntax_penalty,
        )
        reward = breakdown.total
        novelty_bonus = breakdown.reward_semantic_novelty
        policy_reward = reward - novelty_bonus
        self.policy.update(candidate.policy, reward)
        self.prophecy_model.update(
            candidate,
            reward=reward,
            new_kv=new_kv,
            solved_delta=learning_solved_delta,
            status=result.status,
            progress=challenge_progress,
        )
        if result.unavailable:
            self.unavailable_tools.add(candidate.tool_call.tool.value)
        record = StepRecord(
            step=step,
            action=candidate.label,
            template=candidate.template.value,
            what=candidate.policy.what.value,
            how=candidate.policy.how.value,
            where=candidate.policy.where.value,
            status=result.status,
            new_kv=new_kv,
            reward=reward,
            flag_found=flag_found,
            solved_delta=learning_solved_delta,
            solved_total=solved_total,
            imagination_score=imagination_score,
            imagination_support=prediction.support,
            predicted_reward=prediction.expected_reward,
            predicted_solved_rate=prediction.solved_rate,
            novelty_score=novelty_prediction.score,
            novelty_bonus=novelty_bonus,
            novelty_signature=novelty_update.signature,
            policy_reward=policy_reward,
            blocked=result.blocked,
            unavailable=result.unavailable,
            syntax_penalty=syntax_penalty,
            candidate_count=self.last_candidate_count,
            evaluated_candidate_count=self.last_evaluated_candidate_count,
            score_duration_s=self.last_score_duration_s,
            tool_duration_s=result.duration_s,
            policy_sampled=self.last_sampled_policy is not None,
            sampled_policy=_policy_label(self.last_sampled_policy),
            reward_total=reward,
            reward_challenge_solved=breakdown.reward_challenge_solved,
            reward_challenge_progress=breakdown.reward_challenge_progress,
            reward_semantic_novelty=breakdown.reward_semantic_novelty,
            reward_useful_observation=breakdown.reward_useful_observation,
            penalty_repeated_action=breakdown.penalty_repeated_action,
            penalty_repeated_response=breakdown.penalty_repeated_response,
            penalty_invalid_action=breakdown.penalty_invalid_action,
            penalty_no_progress=breakdown.penalty_no_progress,
            challenge_progress=challenge_progress,
            semantic_novelty=novelty_update.semantic_novelty,
            repeated_action=novelty_update.repeated_action,
            repeated_response=novelty_update.repeated_response,
            raw_response_hash=novelty_update.raw_response_hash,
            normalized_response_hash=novelty_update.normalized_response_hash,
            predicted_immediate_solve_probability=prediction.immediate_solve_probability,
            predicted_progress_probability=prediction.progress_probability,
            predicted_eventual_solve_probability=prediction.eventual_solve_probability,
            predicted_expected_progress=prediction.expected_progress,
        )
        self._notify("on_step", record=record, result=result, store=self.store, policy=self.policy, prophecy=self.prophecy_model, dmp=self)
        return record, result

    def _challenge_progress(
        self, *, parsed: list[tuple[KK, str]], flag_found: bool, solved_delta: int,
        meaningful_transition: bool, semantic_novelty: int,
    ) -> float:
        if solved_delta > 0 or flag_found:
            return 1.0
        progress_kinds = {KK.ENDPOINT, KK.AUTH_PATH, KK.SESSION_COOKIE, KK.FLAG, KK.HTTP_METHOD, KK.PARAM_NAME}
        relevant = sum(1 for kk, _ in parsed if kk in progress_kinds)
        if meaningful_transition and relevant:
            return min(0.8, 0.35 + 0.1 * relevant)
        if semantic_novelty > 0 and relevant:
            return min(0.6, 0.15 + 0.08 * relevant)
        return 0.0

    def _reward_breakdown(
        self, *, candidate: ActionCandidate, result: ToolResult, solved_delta: int,
        challenge_progress: float, novelty_update, syntax_penalty: float,
    ) -> RewardBreakdown:
        config = self.reward_config
        failure = result.status == 0 or result.status >= 400 or result.blocked or result.unavailable
        failure_key = f"{novelty_update.signature}:status={result.status}"
        failure_count = self.failure_counts.get(failure_key, 0)
        if failure:
            self.failure_counts[failure_key] = failure_count + 1
        invalid = 0.0
        if result.blocked or result.unavailable or result.status == 0:
            invalid = config.invalid_action
        elif result.status >= 500:
            invalid = 0.2 + min(config.repeated_failure_cap, config.repeated_failure_growth * failure_count)
        elif result.status >= 400:
            invalid = 0.35 + min(config.repeated_failure_cap, config.repeated_failure_growth * failure_count)
        invalid += syntax_penalty if solved_delta == 0 else 0.0
        semantic_reward = min(
            config.semantic_novelty_cap,
            novelty_update.semantic_novelty * (config.semantic_novelty_unit + 0.1 * self.novelty_reward),
        )
        useful = config.useful_observation if (
            not failure and (novelty_update.semantic_novelty > 0 or novelty_update.meaningful_transition)
        ) else 0.0
        no_progress = config.no_progress if solved_delta == 0 and challenge_progress == 0 and novelty_update.semantic_novelty == 0 else 0.0
        return RewardBreakdown(
            reward_challenge_solved=config.challenge_solved * solved_delta,
            reward_challenge_progress=config.challenge_progress * challenge_progress if solved_delta == 0 else 0.0,
            reward_semantic_novelty=semantic_reward,
            reward_useful_observation=useful,
            penalty_repeated_action=config.repeated_action if novelty_update.repeated_action else 0.0,
            penalty_repeated_response=config.repeated_response if novelty_update.repeated_response else 0.0,
            penalty_invalid_action=invalid,
            penalty_no_progress=no_progress,
        )

    def _result(self, success: bool, steps: int, flag: str | None) -> RunResult:
        if not self._trajectory_finalized:
            self.prophecy_model.finalize_episode(self._replay_start)
            self._trajectory_finalized = True
        return RunResult(
            success=success,
            steps=steps,
            flag=flag,
            solved_challenges=list(self.solved_challenges),
            records=self.records,
            knowledge_rows=self.store.rows(),
        )

    def _notify(self, method: str, **payload: object) -> None:
        if self.observer is None:
            return
        callback = getattr(self.observer, method, None)
        if callback is not None:
            callback(**payload)


def candidate_syntax_penalty(candidate: ActionCandidate) -> float:
    penalty = 0.0
    path = candidate.bindings.get(KK.PATH, "")
    endpoint = candidate.bindings.get(KK.ENDPOINT, "")
    if path and _path_appends_directory_to_file(path):
        penalty = max(penalty, 0.95)
    if endpoint and _path_appends_directory_to_file(endpoint):
        penalty = max(penalty, 0.95)
    if candidate.template.value == "HTTP_POST_LOGIN":
        login_path = candidate.bindings.get(KK.PATH, "").lower()
        if not any(token in login_path for token in ("login", "signin", "auth", "session")):
            penalty = max(penalty, 0.90)
    if candidate.template.value in {"HTTP_POST_COMBO", "HTTP_JSON_POST", "HTTP_JSON_PUT", "HTTP_JSON_PATCH"}:
        param_names = candidate.bindings.get(KK.PARAM_NAME, "")
        if param_names.count(",") >= 7:
            penalty = max(penalty, 0.80)
    return penalty


def _path_appends_directory_to_file(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return False
    for part in parts[:-1]:
        name = part.split("?", 1)[0].split("#", 1)[0]
        if "." in name and not name.endswith(".well-known"):
            return True
    return False


def _policy_label(policy: PolicyView | None) -> str:
    if policy is None:
        return ""
    return f"{policy.what.value}/{policy.how.value}/{policy.where.value}"
