# Reproduction

이 페이지는 **current-generation AASSR을 확인하고 reduced validation을 실행하는 최소 경로**를 정리한다.

> [!IMPORTANT]
> 과거 v0.4 runner와 current-generation runner를 섞지 않는다. 현재 source of truth는 `src/aassr_v2/current_manifest.py`다.

---

# 1. 저장소 준비

Windows PowerShell 예시:

```powershell
cd D:\AASSR
git fetch origin
git checkout agent/imagination-gate-ablation
git pull --ff-only origin agent/imagination-gate-ablation
```

가상환경이 없다면:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

이미 `.venv`를 쓰고 있다면 activate 후 dependency만 확인한다.

---

# 2. Current runtime 확인

Python에서 public builder가 current-generation을 가리키는지 확인:

```powershell
python -c "from aassr_v2 import build_pentest_aassr_core; from aassr_v2.current_manifest import CURRENT_GENERATION_VERSION, LEGACY_COMPONENTS_ACTIVE; print(CURRENT_GENERATION_VERSION); print(LEGACY_COMPONENTS_ACTIVE)"
```

예상 핵심:

```text
aassr-current-generation-v2
()
```

`LEGACY_COMPONENTS_ACTIVE`가 비어 있지 않으면 current run을 시작하지 않는다.

---

# 3. Compile + tests

기본 검증:

```powershell
python -m compileall -q src tests scripts
pytest -q
```

Repaired Imagination runner는 자체적으로 중요한 preflight test를 다시 실행한다.

현재 preflight에는 다음 영역이 포함된다.

- current-generation contract
- repaired relational state
- action surface reconstruction
- multimodal future
- outcome probability
- semantic evaluator
- HTTP status supervision
- probability chance backup
- structural planning
- confidence gate
- local Critic support
- structural root deduplication

---

# 4. CUDA hardware path 확인

현재 Policy DQN, world model, Critic이 요청한 CUDA device에 배치되는지 확인:

```powershell
python scripts\check_current_generation_hardware.py --device cuda:0
```

이 단계는 단순히 `torch.cuda.is_available()`만 보는 검사가 아니다. current-generation에서 accelerator 경로가 실제로 사용되는지 확인하기 위한 gate다.

---

# 5. Repaired Imagination reduced validation

현재 연구에서 사용하는 one-shot reduced runner 예시:

```powershell
python scripts\run_repaired_imagination_final.py `
  --output-dir runs\repaired_imagination_2k_status_v3_seed7 `
  --seed 7 `
  --transitions 2048 `
  --block-target 512 `
  --margin 0.05 `
  --max-level 4 `
  --seed-count 4 `
  --device cuda
```

이 runner의 의도는 다음과 같다.

```text
preflight tests
      |
      v
one 2,048-real-transition AASSR training
      |
      v
freeze checkpoint
     / \
    /   \
OFF eval ON eval
    \   /
     \ /
trace + diagnostics
```

## 중요한 조건

- OFF와 ON을 따로 재학습하지 않는다.
- external reward를 바꾸지 않는다.
- intervention margin은 run 전에 고정한다.
- evaluation 중 persistent learning state를 바꾸지 않는다.
- real-transition budget은 primitive environment action 기준이다.

---

# 6. 출력에서 먼저 볼 파일

run이 완료되면 우선 다음을 확인한다.

```text
<output-dir>/summary.json
```

그리고 trace / diagnostic 산출물에서 다음 항목을 본다.

```text
success / failure / stalled / truncation
Imagination plans
switch candidates
interventions
changed actions
intervention errors
HTTP status prediction quality
local Critic support
root coverage
semantic calibration
```

성공률만 보고 run을 판단하지 않는다.

---

# 7. Current-generation local comparison

PyTorch 쪽 current local suite는 다음 조건을 다룬다.

```text
dqn_raw
dqn_relational
aassr_current_no_imagination
aassr_current_full
```

주 runner:

```text
scripts/run_pentest_current_generation_main.py
```

이 실험에서 AASSR OFF/ON은 same checkpoint여야 한다.

---

# 8. DreamerV3 baseline

Canonical DreamerV3는 **Linux/WSL + JAX/CUDA** 환경에서 별도 process로 실행한다.

주 runner:

```text
scripts/run_dreamerv3_current_baseline.py
```

최종 baseline contract:

```text
upstream      : pinned official danijar/dreamerv3
preset        : dmc_proprio + size1m
train ratio   : 1024
dtype         : bfloat16
JAX platform  : cuda
action space  : 240-way relational categorical vocabulary
```

CPU/debug smoke는 API와 Driver-step cadence 확인용이지 benchmark 성능 결과가 아니다.

---

# 9. Final suite assembly

AASSR/PyTorch artifact와 DreamerV3 artifact가 준비되면:

```text
scripts/assemble_pentest_current_generation_suite.py
```

assembler는 다음 mismatch를 허용하지 않는다.

- research seed
- transition budget
- seed pools
- stage manifest
- final-blind status
- Dreamer upstream pin
- Dreamer preset
- train ratio
- dtype
- JAX platform
- 240-way adapter contract

최종 row:

```text
1. dqn_raw
2. dqn_relational
3. dreamerv3_relational
4. aassr_current_no_imagination
5. aassr_current_full
```

---

# 10. 실험을 시작하면 안 되는 경우

다음 중 하나라도 해당하면 결과를 성능 evidence로 사용하지 않는다.

- preflight regression failure
- legacy component가 current runtime에 활성화됨
- OFF/ON이 서로 다른 AASSR checkpoint
- evaluation 중 learning state mutation
- exact transition budget 불일치
- hidden scenario/future/reward leakage
- Dreamer upstream/config mismatch
- final blind seed가 protocol freeze 전에 소비됨

---

# 11. 빠른 해석 체크리스트

run이 끝났을 때:

```text
[ ] 정확한 transition budget인가?
[ ] training-time Imagination intervention은 의도대로 꺼져 있었나?
[ ] OFF/ON은 same frozen checkpoint인가?
[ ] Critic ready인가?
[ ] local Critic support가 실제로 작동했나?
[ ] Imagination run/intervention이 0으로 붕괴하지 않았나?
[ ] intervention이 성공률을 높였나, 아니면 error만 늘렸나?
[ ] 403/404/429 이후 behavior가 합리적인가?
[ ] root dedup으로 runtime이 개선됐나?
[ ] evaluation persistent state가 동결됐나?
```

다음: **[Current Status](Current-Status)**
