# AASSR 위키

### 처음 보는 사람은 여기부터

- **[처음 읽는 사람을 위한 안내서](Beginner-Guide)**
- **[AASSR 5분 설명](AASSR-in-5-Minutes)**
- **[한국어 중심 용어 안내서](Terminology-Guide)**
- **[연구·개발 영어 해설](Research-Jargon-Guide)**
- [개념 지도](Concept-Index)
- [짧은 용어 사전](Glossary)
- **[홈](Home)**

### 연구 · 무엇을 증명하려는가?

- [희소 보상 문제](Sparse-Reward-Problem)
- [연구 질문](Research-Questions)
- **[연구 질문-증거 연결표](Evidence-Matrix)**
- [연구 구조](Research-Architecture)
- [왜 이런 구조를 선택했나?](Design-Rationale)

### 현재 · 지금 무엇까지 맞는가?

- **[현재 연구 상태](Current-Status)**
- [실험 설계와 결과](Experiments)
- [실험 재현 방법](Reproduction)

### AASSR 핵심 구성요소

- [전체 구조](Core-Architecture)
- [상태 표현](State-Representation)
- [ASEQ — 실제 상태·행동·다음 상태 기록](ASEQ)
- [Policy — 기본 정책 모델](Policy)
- [Knowledge — 이번 문제에서 얻은 지식](Knowledge)
- [Prophecy — 미래 예측 모델](Prophecy)
- [Calibration — 예측 신뢰도 보정](Calibration)
- [Critic — 미래 가치 평가기](Critic)
- [Imagination — 가상 미래 탐색](Imagination)
- [Skills — 성공 절차 재사용](Skills)

### 기초 · 강화학습

- [강화학습](Reinforcement-Learning)
- [MDP와 POMDP — 상태·행동·부분 관측](MDP-and-POMDP)
- [희소 보상과 보상 책임 배분](Sparse-Reward-and-Credit-Assignment)
- [탐색과 활용](Exploration-and-Exploitation)
- [정보 이론과 내재 동기](Information-Theory-and-Intrinsic-Motivation)
- [난이도 조절 학습](Curriculum-Learning)
- [가치 함수와 Bellman 식](Value-Functions-and-Bellman-Equation)
- [Q-learning·DQN·TD](Q-Learning-DQN-and-TD)
- [경험 저장소와 에피소드 경계](Replay-Buffer-and-Episode-Boundaries)

### 기초 · 신경망

- [신경망과 최적화](Neural-Networks-and-Optimization)
- [손실 함수와 데이터 불균형](Loss-Functions-and-Class-Imbalance)
- [GRU와 순차 모델](GRU-and-Sequence-Models)

### 기초 · 세계 모델과 계획

- [모델 기반 강화학습과 세계 모델](Model-Based-RL-and-World-Models)
- [확률·불확실성](Stochasticity-Uncertainty-and-Probability)
- [혼합 분포·앙상블·신뢰도 보정](Mixture-Ensemble-and-Calibration)
- [반사실적 계획과 탐색](Counterfactual-Planning-and-Search)
- [환경 결과 노드와 행동 선택 노드](Chance-and-Decision-Nodes)

### 기초 · 일반화와 Skill

- [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)
- [가치 평가 데이터 근거와 학습 분포 밖(OOD)](Critic-Support-and-OOD)
- [계층형 강화학습과 Skill](Hierarchical-RL-and-Skills)

### 연구 방법과 문서 작성

- [인과성·정보 누출·공정 평가](Causality-Leakage-and-Evaluation)
- [구성요소 제거 실험·비교 실험·재현성](Ablation-Benchmarking-and-Reproducibility)
- [위키 문체 안내 — 아마추어 연구자 기준](Language-Style-Guide)

### 과거 실험과 실패 기록

- **[2026-08-11 Imagination 실패 진단](Historical-Imagination-Diagnostic-2026-08-11)**
- [개발 역사](Development-History)

---

### 상태 표시를 읽는 법

- 🟢 **현재 사용 중** — 실제 현재 코드에서 활성화된 구조
- 🟡 **검증 중** — 구현은 됐지만 최종 성능 주장은 아직 검증 중
- 🔵 **메커니즘 근거 있음** — 특정 기능이 의도한 현상을 만든 증거가 있음
- ⚪ **아직 남음** — 현재 세대의 충분한 최종 증거가 없음
- 🕰️ **과거 기록** — 이전 구조·체크포인트에서 나온 결과

### 실제 실행 구조의 최종 기준

`src/aassr_v2/current_manifest.py`
