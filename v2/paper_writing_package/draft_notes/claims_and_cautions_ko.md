# 주장과 주의사항

논문에서 가장 중요한 것은 과장하지 않는 것이다. 아래 문장을 기준으로
본문/발표자료를 작성하면 안전하다.

## Safe Claims

### Framework Claim

```text
AASSR/APASSR는 Knowledge Storage를 단순한 관측 기록소가 아니라 행동
템플릿의 파라미터 후보를 공급하는 typed parameter pool로 사용한다.
```

```text
제안 구조의 핵심은 KK/KV Knowledge Storage, action template binding,
Prophecy Module, Imagination Cycle, DMP closed loop의 결합이다.
```

### Experiment Claim

```text
테스트한 v2_complex GridWorld 환경에서 C3는 Random, PolicyABC, Q-learning,
DQN_PARTIAL보다 높은 성공률과 낮은 반복률을 보였다.
```

```text
GridWorld는 펜테스팅 실험을 대체하는 것이 아니라, 지식-행동 의존성을
통제된 환경에서 검증하기 위한 추상 benchmark이다.
```

### Ablation Claim

```text
TableProphecyModel이 TransformerProphecyModel보다 안정적이었다. 따라서 현재
성능 향상은 neural architecture 복잡도 때문이라고 보기 어렵다.
```

```text
반복 행동 억제와 오류 회피는 현재 C3 구현에서 핵심적인 역할을 했다.
```

```text
무조건적인 지식 획득 점수는 복잡 환경에서 오히려 방해가 될 수 있었다.
```

## Claims to Avoid

피해야 할 주장:

```text
AASSR는 항상 DQN보다 좋다.
```

이유:

```text
locked_bottleneck 또는 더 큰 학습 budget에서는 DQN/Q-learning이 더 좋게
나올 수 있다.
```

피해야 할 주장:

```text
GridWorld 결과가 실제 펜테스팅 성능을 증명한다.
```

이유:

```text
GridWorld는 추상 검증 환경이다. 실제 nmap 기반 펜테스팅 실험은 future work
또는 별도 실험으로 두어야 한다.
```

피해야 할 주장:

```text
현재 framework는 Transformer 기반 방법이다.
```

이유:

```text
TransformerProphecyModel은 ablation용 구현 변형이다. 메인 framework는 C3이며
현재 C3는 TableProphecyModel을 사용한다.
```

피해야 할 주장:

```text
Imagination Cycle이 실제 미래 환경을 rollout한다.
```

이유:

```text
현재 Imagination은 hidden map을 읽거나 미래 행동을 실제 실행하지 않는다.
Prophecy prediction 기반 후보 평가이다.
```

## Strong But Safe Framing

가장 좋은 중심 문장:

```text
AASSR의 강점은 모든 task에서 범용 RL을 이기는 데 있는 것이 아니라,
지식이 다음 행동의 파라미터가 되는 구조적 의존성 task에서 반복과 오류를
줄이며 더 효율적으로 탐색하는 데 있다.
```

영어 버전:

```text
The strength of AASSR is not that it universally dominates general-purpose RL,
but that it explicitly models tasks where discovered knowledge becomes the
parameter source for future actions, enabling more efficient exploration by
reducing repeated and error-prone candidates.
```

## Handling DQN Results

DQN이 더 좋은 결과가 나왔을 때:

```text
This does not invalidate AASSR. Instead, it clarifies the scope of the method:
AASSR is intended for knowledge-action dependency settings where interpretability,
sample efficiency, and action parameter reuse matter.
```

한국어:

```text
DQN이 일부 환경에서 더 좋은 것은 AASSR의 실패가 아니라, AASSR의 적용
범위를 명확히 해주는 결과이다. AASSR는 지식 재사용, 행동 파라미터화,
반복/오류 감소가 중요한 환경에서 강점을 갖는다.
```

## Most Defensible Conclusion

```text
본 연구는 Knowledge Storage를 행동 생성의 파라미터 공급원으로 재해석하고,
Prophecy와 Imagination을 통해 실행 전 후보 행동을 평가하는 폐루프 구조를
제시했다. GridWorld 추상 실험은 이 구조가 지식-행동 의존성이 있는 환경에서
반복 행동과 오류를 줄이며 탐색 효율을 높일 수 있음을 보여준다.
```
