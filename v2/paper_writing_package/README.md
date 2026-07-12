# AASSR/APASSR Paper Writing Package

이 폴더는 AASSR/APASSR 논문 초안을 작성하기 위한 독립 패키지이다.
원본 실행 결과와 핵심 코드 참조본을 한 곳에 모아두었다.

## Folder Structure

```text
paper_writing_package/
├─ README.md
├─ draft_notes/
│  ├─ paper_outline_ko.md
│  ├─ claims_and_cautions_ko.md
│  ├─ experiment_summary_ko.md
│  └─ code_reference_guide_ko.md
├─ data/
│  ├─ main_results/
│  └─ ablations/
├─ figures/
│  ├─ main_results/
│  └─ worlds/
└─ code_reference/
```

## Recommended Reading Order

1. `draft_notes/paper_outline_ko.md`
   - 논문 전체 구조와 각 섹션에 들어갈 내용.

2. `draft_notes/experiment_summary_ko.md`
   - main comparison, ablation 결과, 해석.

3. `draft_notes/claims_and_cautions_ko.md`
   - 논문에서 안전하게 주장할 수 있는 것과 피해야 할 것.

4. `draft_notes/code_reference_guide_ko.md`
   - 어떤 코드 파일이 어떤 논문 모듈에 대응되는지.

5. `data/` and `figures/`
   - 실제 표/그림/보고서 원본.

## Core Thesis

```text
AASSR/APASSR는 단순한 강화학습 정책이 아니라,
관측으로 얻은 지식을 다음 행동 템플릿의 파라미터로 재사용하는
knowledge-parameterized closed-loop decision-making framework이다.
```

핵심 문장:

```text
행동이 지식을 만들고, 지식이 다음 행동을 만든다.
```

## Main Paper Condition

논문 본문에서 중심으로 둘 조건은 C3이다.

```text
C3 = PolicyABC + Prophecy Module + Imagination Cycle
```

현재 C3 구현은 `TableProphecyModel`을 사용한다. 그러나 논문의 핵심 기여는
table 모델 자체가 아니라 다음 폐루프 구조이다.

```text
Knowledge Storage
-> KK/KV action parameter binding
-> PolicyABC
-> Prophecy Module
-> Imagination Cycle
-> Execution
-> Observation
-> Knowledge update
```

## Important Scope Note

GridWorld는 원래 nmap 기반 펜테스팅 실험을 대체하는 것이 아니다.
펜테스팅에서 나타나는 지식-행동 의존성을 통제된 환경에서 검증하기 위한
추상 benchmark이다.

## Data Included

`data/main_results/`:

- `v2_complex_30x10_summary_table.csv`
- `v2_complex_30x10_condition_stats.csv`
- `v2_complex_30x10_report.md`
- `locked_bottleneck_30x10_summary_table.csv`
- `locked_bottleneck_30x10_condition_stats.csv`
- `locked_bottleneck_30x10_report.md`

`data/ablations/`:

- ablation 1: Table vs Transformer Prophecy
- ablation 2: Prophecy prediction-error reward ON/OFF
- ablation 3: Imagination depth/branch sweep
- ablation 4: Imagination mechanism terms
- ablation 5: Prophecy score components
- reward-off 100x10 check

## Figures Included

`figures/main_results/`:

- success rate
- steps to flag
- semantic gain
- repeat/error rate
- learning curve

`figures/worlds/`:

- random_key_door environment render
- v2_complex environment render
- locked_bottleneck environment render

## Safe One-Sentence Summary

```text
In controlled GridWorld environments designed to mimic knowledge-action
dependencies, C3 improved success and reduced repeated actions relative to
random, simple PolicyABC, Q-learning, and a partial-observation DQN baseline in
the tested v2_complex setting; ablation studies suggest that repeat suppression
and error avoidance are especially important in the current implementation.
```
