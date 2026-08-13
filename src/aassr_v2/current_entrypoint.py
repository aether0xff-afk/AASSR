from __future__ import annotations

from dataclasses import replace

from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_confidence_gate import install_current_confidence_gate
from .current_critic_support import install_critic_support_gate
from .current_decision_optimization import install_current_decision_optimizations
from .current_hardware import install_hardware_dqn
from .current_hot_path_profile import install_current_hot_path_profiler
from .current_manifest import CURRENT_COMPONENTS
from .current_planner import CurrentFullyBatchedImaginationTree
from .current_plugin_api import CurrentRuntimePlugin, bind_current_core_plugin_boundary
from .current_repair import install_current_repairs
from .current_root_dedup import install_structural_root_dedup
from .current_runtime_performance_policy import (
    install_current_runtime_performance_device_aware,
)
from .current_validation import install_current_fast_validation
from .plugins.current_pentest import CURRENT_PENTEST_PLUGIN


# The repaired 2k diagnostic established this gate threshold. Scaling the training
# budget must not silently change the intervention criterion at the same time.
CURRENT_INTERVENTION_MARGIN = 0.05


def _enable_current_full_batching(agent: CurrentStandalonePentestAASSRAgent) -> None:
    old = agent.planner
    planner = CurrentFullyBatchedImaginationTree(
        old.policy,
        agent.current_batched_prophecy,
        config=old.config,
        scorer=old.scorer,
        representation=getattr(agent, "representation", None),
    )
    planner._state_key = old._state_key
    agent.planner = planner
    agent.core.planner = planner
    agent.current_depth_batching = True
    agent.current_critic_batching = True


def build_current_aassr_core(
    *,
    plugin: CurrentRuntimePlugin,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    enable_batching: bool = True,
    allow_tf32: bool = True,
    profile_hot_path: bool = False,
    enable_performance_optimizations: bool = True,
) -> CurrentStandalonePentestAASSRAgent:
    """Assemble the current AASSR core with one explicit runtime plugin.

    The algorithmic AASSR core owns Policy, Prophecy/Imagination roles, Critic,
    Knowledge, Skills, ASEQ and reliability gates. ``CURRENT_PENTEST_PLUGIN`` owns
    the response-causal observation contract, relational HTTP/action binding,
    categorical public-status supervision and pentest environment semantics.

    Historical builders remain importable only for reproduction; current pentest
    runners should resolve through this function.

    ``enable_performance_optimizations`` changes only implementation mechanics:
    indexing, redundant state encoding, accelerator synchronization and tensor
    packing. It does not alter seeds, replay samples, update cadence, batch size,
    losses, exploration or action/value semantics. CPU keeps only safe Python
    fast paths; accelerator-specific fusion/packing is enabled only on CUDA.
    """

    if not enable_batching:
        raise ValueError(
            "current-generation pentest AASSR requires semantics-preserving "
            "Prophecy/Critic depth batching; use an explicit legacy/research "
            "entrypoint for historical scalar reproduction"
        )

    agent = build_current_standalone_pentest_aassr_core(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
        representation=plugin.representation,
    )
    agent.config = replace(
        agent.config,
        imagination_intervention_margin=CURRENT_INTERVENTION_MARGIN,
    )
    _enable_current_full_batching(agent)
    hardware = install_hardware_dqn(
        agent,
        seed=int(seed),
        train_transitions=int(train_transitions),
        device=device,
        allow_tf32=bool(allow_tf32),
    )

    install_current_fast_validation(agent)
    install_current_repairs(
        agent,
        seed=int(seed),
        device=hardware.resolved_device,
    )
    plugin.install_world_model(
        agent,
        seed=int(seed),
        device=hardware.resolved_device,
    )
    if enable_performance_optimizations:
        install_current_runtime_performance_device_aware(agent)
    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    agent.current_components = dict(CURRENT_COMPONENTS)
    install_current_confidence_gate(agent)
    install_critic_support_gate(agent)
    install_structural_root_dedup(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    bind_current_core_plugin_boundary(agent, plugin)
    return agent


def build_current_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    enable_batching: bool = True,
    allow_tf32: bool = True,
    profile_hot_path: bool = False,
    enable_performance_optimizations: bool = True,
) -> CurrentStandalonePentestAASSRAgent:
    """Canonical pentest assembly; domain behavior comes only from its plugin."""

    return build_current_aassr_core(
        plugin=CURRENT_PENTEST_PLUGIN,
        seed=seed,
        train_transitions=train_transitions,
        use_imagination=use_imagination,
        device=device,
        enable_batching=enable_batching,
        allow_tf32=allow_tf32,
        profile_hot_path=profile_hot_path,
        enable_performance_optimizations=enable_performance_optimizations,
    )
