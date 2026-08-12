# `aassr_v2` 소스 구조

이 패키지에는 현행 runtime과 과거 연구 구현이 함께 있습니다. **현행 여부는 파일의 생성 시점이 아니라 import 경로로 판단합니다.**

## 현행 핵심

- `current_manifest.py` — 활성 component 계약
- `current_entrypoint.py` — 유일한 canonical pentest builder
- `current_agent.py` — current standalone agent
- `current_relational_state_v3.py` — 공개 relational state v3
- `current_status_models.py` — status-balanced conditional-mixture Prophecy
- `current_return_critic.py` — sparse-return Critic
- `current_critic_support.py` — local real-training OOD/support gate
- `current_confidence_gate.py` — prediction reliability gate
- `current_root_dedup.py` — structural root compute dedup
- `current_planner.py` — current batched Imagination planner
- `current_decision_optimization.py` — current decision hot path
- `pentest_current_generation_main.py` — canonical 학습/평가 protocol

## 환경/프로토콜

- `pentest_transfer_stages.py` — transfer stages와 adaptive curriculum
- `pentest_curriculum_env.py` — in-process pentest environment
- `current_protocol.py` — current episode/evaluation protocol

## Baseline

- `current_dqn_baseline.py` — Raw/Relational DQN controls
- `dreamerv3_baseline.py`, `dreamerv3_external.py`, `dreamerv3_hardware.py` — official DreamerV3 비교 경로

## 과거 연구 구현

v0.4, Neural Delta 이전 세대, GridPush, ToolGrid, Imagination v2, paper experiment 관련 모듈은 재현성을 위해 남아 있습니다. 새 현행 코드에서 이들을 암묵적으로 다시 활성화하면 안 됩니다.

최종 활성 목록은 항상 `CURRENT_COMPONENTS`와 `LEGACY_COMPONENTS_ACTIVE`를 확인하세요.
