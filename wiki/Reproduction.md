# 실험 재현 방법 (Reproduction)

이 페이지는 **현재 `main`의 AASSR current-generation을 재현하고, 결과를 연구 evidence로 사용할 수 있는지 확인하는 최소 절차**를 설명한다.

> [!IMPORTANT]
> Current architecture의 [최종 기준(source of truth)](Current-Status)는 특정 연구 브랜치가 아니라 `main`의 `src/aassr_v2/current_manifest.py`다. 과거 실험을 재현할 때만 해당 historical commit/branch를 명시적으로 checkout한다.

관련 페이지:
- [Current Status](Current-Status)
- [Experiments](Experiments)
- [Evidence Matrix](Evidence-Matrix)
- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

## 목차

1. [Current source checkout](#1-current-source-checkout)
2. [Python environment](#2-python-environment)
3. [Manifest 확인](#3-manifest-확인)
4. [Regression gate](#4-regression-gate)
5. [CUDA path 확인](#5-cuda-path-확인)
6. [Reduced same-checkpoint validation](#6-reduced-same-checkpoint-validation)
7. [Current local comparison](#7-current-local-comparison)
8. [DreamerV3 baseline](#8-dreamerv3-baseline)
9. [Final suite assembly](#9-final-suite-assembly)
10. [결과를 evidence로 쓰면 안 되는 경우](#10-결과를-evidence로-쓰면-안-되는-경우)
11. [Historical reproduction](#11-historical-reproduction)

---

# 1. Current source checkout

Windows PowerShell:

```powershell
cd D:\AASSR

git fetch origin
git checkout main
git pull --ff-only origin main
```

실험 보고서에는 가능하면 실제 실행 commit을 기록한다.

```powershell
git rev-parse HEAD
```

예:

```text
AASSR commit: <40-char SHA>
```

왜 필요한가?

```text
“2026-08-12의 AASSR”
```

보다:

```text
“commit abcdef...의 AASSR”
```

가 훨씬 정확하게 재현 가능하기 때문이다.

> [!TIP]
> 새 실험을 시작한 뒤 `main`이 바뀌어도 같은 run을 재현하려면 시작 commit SHA를 보존한다.

---

# 2. Python environment

가상환경이 없다면:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

이미 `.venv`가 있다면:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

GPU-specific optional dependency가 필요한 runner라면 저장소의 current dependency contract에 맞춰 설치한다.

환경 기록 권장:

```powershell
python --version
python -m pip freeze > runs\environment-freeze.txt
```

CUDA/PyTorch 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

# 3. Manifest 확인

Current runtime 이름과 legacy component 여부:

```powershell
python -c "from aassr_v2.current_manifest import CURRENT_GENERATION_VERSION, CURRENT_COMPONENTS, LEGACY_COMPONENTS_ACTIVE; print(CURRENT_GENERATION_VERSION); print(LEGACY_COMPONENTS_ACTIVE); [print(f'{k}: {v}') for k,v in CURRENT_COMPONENTS.items()]"
```

현재 핵심 기대값:

```text
aassr-current-generation-v2
LEGACY_COMPONENTS_ACTIVE = ()
```

그리고 current manifest에는 최소한 다음 계열이 보여야 한다.

```text
observation
  response-causal-relational-public-state-v3+latest-http-status

prophecy
  relational-conditional-mixture-ensemble-v5-status-balanced

calibration
  semantic-probability-holdout-calibration-v3-status-aware

critic_support_gate
  local-real-training-support-fail-closed-v1

training_imagination
  disabled-same-checkpoint
```

`LEGACY_COMPONENTS_ACTIVE`가 비어 있지 않거나 manifest가 문서와 다르면 **먼저 [Current Status](Current-Status)를 갱신한 뒤 실험을 해석**한다.

---

# 4. Regression gate

가장 넓은 기본 검증:

```powershell
python -m compileall -q src tests scripts
pytest -q
```

긴 전체 test가 부담되면 current-generation runner가 사용하는 preflight subset을 우선 사용할 수 있지만, 성능 evidence를 만들기 전에는 current contract에 직접 관련된 [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)이 모두 통과해야 한다.

특히 확인할 영역:

- [State Representation v3](State-Representation)
- public HTTP status preservation
- [행동(action)](Reinforcement-Learning)-surface reconstruction
- [Prophecy](Prophecy) multimodal outcome preservation
- [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability) normalization
- status categorical supervision
- [Calibration](Calibration)
- [Chance vs Decision](Chance-and-Decision-Nodes) backup
- [Critic](Critic) sparse-[누적 보상(return)](Value-Functions-and-Bellman-Equation) contract
- [Local Critic Support](Critic-Support-and-OOD)
- structural root deduplication
- [같은 체크포인트(same-checkpoint)](Experiments) freeze

Regression pass는 **성능 향상 evidence가 아니라 구현 contract evidence**다.

---

# 5. CUDA path 확인

현재 hardware path 검사:

```powershell
python scripts\check_current_generation_hardware.py --device cuda:0
```

이 단계의 목적은 단순히:

```text
torch.cuda.is_available() == True
```

를 확인하는 것이 아니다.

[Policy(정책 모델)](Policy) [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD), relational [세계 모델(world model)](Model-Based-RL-and-World-Models), [Critic(미래 가치 평가기)](Critic)과 current [묶음 처리(batching)](Reproduction) path가 **실제로 요청한 accelerator contract를 사용하고 있는지** 확인한다.

성능 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)에서 CPU fallback이 일어났다면 runtime 비교를 그대로 사용하면 안 된다.

---

# 6. Reduced same-checkpoint validation

현재 저장소에는 repaired [Imagination(가상 미래 탐색)](Imagination) validation entrypoint가 있다.

예시:

```powershell
python scripts\run_repaired_imagination_final.py `
  --output-dir runs\repaired_imagination_current_seed7 `
  --seed 7 `
  --transitions 2048 `
  --block-target 512 `
  --margin 0.05 `
  --max-level 4 `
  --seed-count 4 `
  --device cuda
```

> [!NOTE]
> 위 `2048`, `512`, `0.05`, `4`는 **reduced diagnostic 예시**다. 이것을 final 표준 비교 실험 budget과 혼동하지 않는다. 실제 논문/보고서 결과에는 실행 argument 전체와 commit SHA를 함께 기록한다.

핵심 구조:

```text
real training
     ↓
one AASSR checkpoint
     ↓ freeze
   /          \
OFF            ON
Policy       Imagination
   \          /
    same evaluation protocol
```

## 반드시 확인할 것

- OFF/ON을 따로 재학습하지 않았는가?
- evaluation 중 persistent learner state가 변하지 않았는가?
- external [보상(reward)](Sparse-Reward-and-Credit-Assignment) contract가 동일한가?
- [실제 행동 개입(intervention)](Imagination) margin을 결과 보기 전에 고정했는가?
- [상태 전이(transition)](MDP-and-POMDP) budget이 real primitive 행동 기준인가?

---

# 7. Current local comparison

PyTorch current-generation local comparison entrypoint:

```text
scripts/run_pentest_current_generation_main.py
```

핵심 condition:

```text
dqn_raw
dqn_relational
aassr_current_no_imagination
aassr_current_full
```

각 비교의 의미:

```text
dqn_raw
→ dqn_relational
= representation effect

dqn_relational
→ AASSR no-Imagination
= non-Imagination AASSR stack effect

AASSR no-Imagination
→ AASSR Full
= Imagination marginal effect
```

세부 설계: [Experiments](Experiments)

---

# 8. DreamerV3 baseline

External model-based [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility) entrypoint:

```text
scripts/run_dreamerv3_current_baseline.py
```

Canonical 표준 비교 실험는 Linux/WSL + JAX/CUDA 환경의 pinned official upstream을 사용한다.

최종 결과에 기록해야 할 것:

```text
upstream commit / pin
preset
train ratio
dtype
JAX platform
action adapter contract
real primitive transition budget
```

CPU/debug smoke는 API와 [환경(environment)](Reinforcement-Learning) stepping contract 확인용이다.

```text
CPU smoke success
!=
canonical CUDA benchmark result
```

외부 비교 기준의 알고리즘 내부를 AASSR에 맞춰 임의로 수정하면 비교 의미가 달라지므로 변경이 있다면 반드시 별도 condition으로 이름을 바꾼다.

---

# 9. Final suite assembly

현재 suite assembly entrypoint:

```text
scripts/assemble_pentest_current_generation_suite.py
```

최종 목표 row:

```text
1. dqn_raw
2. dqn_relational
3. dreamerv3_relational
4. aassr_current_no_imagination
5. aassr_current_full
```

Assembler 또는 사람이 확인해야 하는 주요 mismatch:

- source commit
- research [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)
- 상태 전이 budget
- train/eval 난수 시드 pools
- stage/tier protocol
- final-blind 사용 여부
- AASSR 같은 체크포인트 여부
- Dreamer upstream/config
- [관측(observation)](MDP-and-POMDP) [표현(representation)](Relational-Representation-and-Generalization) version
- 보상 contract

---

# 10. 결과를 evidence로 쓰면 안 되는 경우

다음 중 하나라도 해당하면 숫자가 나와도 **current performance evidence로 승격하지 않는다.**

```text
[ ] regression failure
[ ] legacy component active
[ ] hidden simulator leakage
[ ] OFF/ON different checkpoint
[ ] evaluation learning-state mutation
[ ] transition budget mismatch
[ ] representation version mismatch
[ ] reward contract mismatch
[ ] CPU fallback while claiming CUDA runtime
[ ] final blind set consumed before protocol freeze
[ ] external baseline config mismatch
```

또한:

```text
한 seed의 diagnostic
```

을:

```text
일반적인 성능 우위
```

로 표현하지 않는다.

관련: [Evidence Matrix](Evidence-Matrix), [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)

---

# 11. Historical reproduction

과거 결과를 재현할 때는 current `main`으로 억지로 재현하지 않는다.

예:

```text
2026-08-11 historical Imagination diagnostic
```

을 재현하려면 해당 당시 commit/branch와 당시 표현/[Prophecy(미래 예측 모델)](Prophecy) contract를 사용해야 한다.

그 결과를 current performance와 분리해 저장한다.

대표 사례:

- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)
- [Development History](Development-History)

---

## Run report template

실험 결과와 함께 최소한 다음 metadata를 남긴다.

```text
AASSR commit:
current generation:
research seed:
scenario/eval seed pool:
training transitions:
device:
reward contract:
representation contract:
Prophecy contract:
Calibration contract:
Critic support contract:
Imagination margin:
max level / horizon:
OFF/ON same checkpoint verified:
evaluation frozen verified:
```

Result:

```text
success:
true failure:
stalled:
truncation:
mean requests:
planner plans:
final interventions:
bad-status interventions:
local-support rejects:
runtime:
```

이 metadata가 있으면 나중에 숫자의 출처와 의미를 훨씬 쉽게 복구할 수 있다.

---

## 다음으로 읽기

- [Current Status](Current-Status)
- [Experiments](Experiments)
- [Evidence Matrix](Evidence-Matrix)
- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)
