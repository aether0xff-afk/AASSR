# 실험 결과 요약

이 문서는 논문 Results/Ablation 섹션에 바로 옮길 수 있는 형태로 현재 결과를
정리한다.

## 1. Main Comparison

주요 참조 파일:

```text
data/main_results/v2_complex_30x10_summary_table.csv
data/main_results/v2_complex_30x10_report.md
data/main_results/locked_bottleneck_30x10_summary_table.csv
data/main_results/locked_bottleneck_30x10_report.md
```

### v2_complex 30x10

| Condition | Success rate | Repeat rate | Interpretation |
| --- | ---: | ---: | --- |
| C3 | 0.717 | 0.254 | main APASSR condition |
| DQN_PARTIAL | 0.593 | 0.354 | partial-observation DQN baseline |
| QLEARN | 0.573 | 0.372 | tabular Q-learning baseline |
| C1 | 0.540 | about 0.35 | PolicyABC only |
| C2 | 0.530 | about 0.35 | Prophecy reward without Imagination |
| C0 | 0.427 | 0.504 | random baseline |

핵심 해석:

```text
C3는 v2_complex에서 QLEARN 및 DQN_PARTIAL보다 높은 성공률과 낮은 반복률을
보였다. 이는 단순히 더 많이 시도해서 성공한 것이 아니라, 반복 행동을
줄이며 탐색 효율을 높였다는 뜻이다.
```

주의:

```text
ORACLE_MDP는 전체 지도를 아는 upper bound이므로 C3와 같은 조건 baseline이 아니다.
```

### locked_bottleneck 30x10

locked_bottleneck에서는 QLEARN/DQN 계열이 더 강하게 나올 수 있다. 이는
AASSR가 모든 환경에서 항상 DQN보다 좋다는 주장을 피해야 함을 보여준다.

안전한 해석:

```text
AASSR의 장점은 모든 환경에서의 지배적 성능이 아니라, 지식 재사용과
반복/오류 감소가 중요한 구조적 환경에서의 효율적 탐색이다.
```

## 2. Ablation 1: Table vs Transformer Prophecy

참조 파일:

```text
data/ablations/*ablation_1_prophecy_model*summary_table.csv
```

| Environment | Table C3 | Transformer C3 | Interpretation |
| --- | ---: | ---: | --- |
| random_key_door | 0.970 | 0.880 | Table more stable |
| v2_complex | 0.717 | 0.680 | Table better success |
| locked_bottleneck | 0.423 | 0.397 | Table slightly better |

논문 해석:

```text
성능 향상은 더 복잡한 neural Prophecy 구조 때문이 아니다. 현재 규모에서는
TableProphecyModel이 더 sample-efficient하고 안정적이었다.
```

## 3. Ablation 2: Prophecy Reward ON/OFF

참조 파일:

```text
data/ablations/*ablation_2_prophecy_reward*summary_table.csv
data/ablations/*ablation_reward_off_check_100x10*summary_table.csv
```

30x10 결과:

| Environment | Reward ON | Reward OFF | Interpretation |
| --- | ---: | ---: | --- |
| random_key_door | 0.970 | 0.973 | saturated simple task |
| v2_complex | 0.717 | 0.700 | ON slightly better |
| locked_bottleneck | 0.423 | 0.370 | ON meaningfully better |

100x10 재확인:

| Environment | Reward ON | Reward OFF |
| --- | ---: | ---: |
| random_key_door | 0.964 | 0.970 |
| v2_complex | 0.726 | 0.713 |
| locked_bottleneck | 0.405 | 0.357 |

해석:

```text
Prediction-error reward는 핵심 성능 원인은 아니지만, 복잡하거나 dependency-heavy한
환경에서는 약한 보조 신호로 도움이 된다.
```

## 4. Ablation 3: Imagination Depth/Branch

참조 파일:

```text
data/ablations/*ablation_3_imagination_depth_branch*summary_table.csv
```

v2_complex:

| Setting | Success |
| --- | ---: |
| D2_B1 | 0.737 |
| D3_B1 | 0.723 |
| D2_B3 | 0.717 |
| D1_B1 | 0.703 |

locked_bottleneck:

| Setting | Success |
| --- | ---: |
| D3_B1 | 0.493 |
| D1_B1 | 0.470 |
| D2_B1 | 0.453 |
| D2_B3 | 0.423 |

해석:

```text
얕은 depth-limited Imagination은 도움이 되었지만, branch를 넓게 늘리는 것은
일관된 성능 향상을 만들지 않았다.
```

## 5. Ablation 4: Imagination Mechanisms

참조 파일:

```text
data/ablations/*ablation_4_imagination_mechanisms*summary_table.csv
```

핵심 발견:

```text
NO_REPEAT_PENALTY가 크게 망했다.
```

| Environment | Full C3 | No Repeat Penalty |
| --- | ---: | ---: |
| random_key_door | 0.970 | 0.880 |
| v2_complex | 0.717 | 0.383 |
| locked_bottleneck | 0.423 | 0.077 |

해석:

```text
현재 C3 성능에서 반복 행동 억제는 핵심 부품이다.
```

추가 발견:

```text
NO_POLICY_PRIOR가 v2_complex와 locked_bottleneck에서 Full C3보다 좋았다.
```

| Environment | Full C3 | No Policy Prior |
| --- | ---: | ---: |
| v2_complex | 0.717 | 0.730 |
| locked_bottleneck | 0.423 | 0.530 |

해석:

```text
C3 성능이 PolicyABC prior에 기대서 나온 것은 아니다. 복잡 환경에서는
policy prior가 오히려 방해될 수 있다.
```

## 6. Ablation 5: Prophecy Score Components

참조 파일:

```text
data/ablations/*ablation_5_prophecy_score_components*summary_table.csv
```

핵심 발견 1:

```text
NO_KNOWLEDGE_GAIN이 오히려 좋아졌다.
```

| Environment | Full C3 | No Knowledge Gain |
| --- | ---: | ---: |
| random_key_door | 0.970 | 0.923 |
| v2_complex | 0.717 | 0.750 |
| locked_bottleneck | 0.423 | 0.763 |

해석:

```text
복잡 환경에서 현재 knowledge_weight는 너무 강해서 목표 달성보다 새 지식 수집에
과하게 끌렸을 수 있다.
```

핵심 발견 2:

```text
NO_ERROR_AVOIDANCE가 복잡 환경에서 크게 나빠졌다.
```

| Environment | Full C3 | No Error Avoidance |
| --- | ---: | ---: |
| v2_complex | 0.717 | 0.683 |
| locked_bottleneck | 0.423 | 0.147 |

해석:

```text
오류 회피는 복잡한 문/벽/반복 구조에서 매우 중요하다.
```

## 7. Results Section에 쓸 수 있는 핵심 문장

```text
The main comparison shows that C3 improves success rate and reduces repeated
actions in the v2_complex environment compared with Random, PolicyABC,
Q-learning, and DQN_PARTIAL baselines.
```

```text
The ablation studies suggest that the current implementation benefits most from
repeat suppression and error avoidance rather than from indiscriminate
knowledge-gain seeking.
```

```text
TableProphecyModel outperformed the Transformer implementation in the tested
settings, indicating that the framework's benefit is not simply due to neural
model complexity.
```

```text
Removing knowledge-gain scoring improved performance in the more structured
environments, suggesting that the agent should prioritize actionable and safe
knowledge rather than maximizing the amount of newly discovered information.
```
