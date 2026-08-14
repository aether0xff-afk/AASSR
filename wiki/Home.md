# AASSR

AASSR은 **희소 보상 환경에서 에이전트가 정답 전략을 직접 주입받지 않고 실제 경험으로 의미와 관계를 배우고, 미래를 예측하여 행동을 선택할 수 있는가**를 연구하는 강화학습 아키텍처다.

## 현재 가장 중요한 설계 원칙

> **Plugin은 세계의 문법만 알려주고, 세계의 의미는 AASSR Core가 스스로 배운다.**

이 원칙에 따라 2026-08-14부터 Core와 환경 연결부를 다시 분리하고 있다.

```text
실제 환경
  ↓
Plugin
(행동 문법 / 공개 정보 종류 / 실제 I/O)
  ↓
AASSR Core
(표현 / Knowledge / Policy / Prophecy / Critic / Imagination / ASEQ / Skills)
```

## 처음 읽는 순서

1. [[AASSR in 5 Minutes|AASSR-in-5-Minutes]]
2. [[Core Architecture|Core-Architecture]]
3. [[플러그인 제작법|Plugin-Development]]
4. [[현재 상태|Current-Status]]
5. [[실험|Experiments]]
6. [[증거 행렬|Evidence-Matrix]]

## 과거 pentest 실험

기존 10k pentest runtime과 checkpoint는 재현성을 위해 보존한다. 그러나 그 runtime의 환경 전용 표현과 Plugin 권한은 새 Core 설계 기준이 아니다. 과거 수치는 historical evidence로 분리하고 새 Core 성능은 다시 측정한다.
