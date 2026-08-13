from __future__ import annotations

import re
from pathlib import Path

WIKI = Path("wiki")

# High-frequency bare English found by audit_wiki_bare_english.py after the first
# Korean-first passes. These are ordinary words to specialists but real barriers
# for an amateur developer or student researcher reading a long technical page.
TERMS: dict[str, tuple[str, str]] = {
    "task": ("[연구 과제(task)](Sparse-Reward-Problem)", "연구 과제"),
    "target": ("[대상 또는 학습 목표값(target)](Terminology-Guide)", "대상/목표값"),
    "calibration": ("[예측 신뢰도 보정(calibration)](Calibration)", "예측 신뢰도 보정"),
    "sample": ("[표본(sample)](Ablation-Benchmarking-and-Reproducibility)", "표본"),
    "future": ("[미래(future)](Counterfactual-Planning-and-Search)", "미래"),
    "decision": ("[의사결정(decision)](Chance-and-Decision-Nodes)", "의사결정"),
    "trajectory": ("[경험 경로(trajectory)](Reinforcement-Learning)", "경험 경로"),
    "sequence": ("[순서열(sequence)](GRU-and-Sequence-Models)", "순서열"),
    "rate": ("[비율(rate)](Terminology-Guide)", "비율"),
    "error": ("[오차(error)](Loss-Functions-and-Class-Imbalance)", "오차"),
    "level": ("[난이도 단계(level)](Curriculum-Learning)", "난이도 단계"),
    "count": ("[횟수(count)](Terminology-Guide)", "횟수"),
    "exact": ("[정확히 동일한(exact)](ASEQ)", "정확히 동일한"),
    "mixture": ("[여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration)", "혼합 분포"),
    "information": ("[정보(information)](Information-Theory-and-Intrinsic-Motivation)", "정보"),
    "class": ("[범주(class)](Loss-Functions-and-Class-Imbalance)", "범주"),
    "budget": ("[실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)", "실험 예산"),
    "learning": ("[학습(learning)](Reinforcement-Learning)", "학습"),
    "accuracy": ("[정확도(accuracy)](Ablation-Benchmarking-and-Reproducibility)", "정확도"),
    "Planner": ("[계획기(Planner)](Counterfactual-Planning-and-Search)", "계획기"),
    "planner": ("[계획기(planner)](Counterfactual-Planning-and-Search)", "계획기"),
    "quality": ("[품질(quality)](Ablation-Benchmarking-and-Reproducibility)", "품질"),
    "uncertainty": ("[불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability)", "불확실성"),
    "structure": ("[구조(structure)](Research-Architecture)", "구조"),
    "latest": ("[가장 최근의(latest)](Current-Status)", "가장 최근의"),
    "primitive": ("[더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills)", "기본 행동 단위"),
    "Relational": ("[관계 기반(Relational)](Relational-Representation-and-Generalization)", "관계 기반"),
    "Current": ("[현재(Current)](Current-Status)", "현재"),
    "raw": ("[가공하지 않은 원본(raw)](State-Representation)", "원본"),
    "replay": ("[저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)", "경험 재사용"),
    "final": ("[최종(final)](Ablation-Benchmarking-and-Reproducibility)", "최종"),
    "Training": ("[학습(Training)](Reinforcement-Learning)", "학습"),
    "fact": ("[실제로 관측한 사실(fact)](Causality-Leakage-and-Evaluation)", "실제 관측 사실"),
    "stalled": ("[진전 없이 반복하다 멈춘(stalled)](ASEQ)", "진전 없이 멈춘"),
    "sparse": ("[드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment)", "희소한"),
    "data": ("[데이터(data)](Terminology-Guide)", "데이터"),
    "legal": ("[현재 허용된(legal)](Terminology-Guide)", "현재 허용된"),
    "surface": ("[현재 선택 가능한 영역(surface)](Terminology-Guide)", "선택 가능 영역"),
    "shaping": ("[인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)", "형태 조정"),
    "memory": ("[기억(memory)](GRU-and-Sequence-Models)", "기억"),
    "semantics": ("[의미 규칙(semantics)](State-Representation)", "의미 규칙"),
    "depth": ("[탐색 깊이(depth)](Counterfactual-Planning-and-Search)", "탐색 깊이"),
    "rollout": ("[가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)", "가상 미래 전개"),
    "update": ("[학습 갱신(update)](Neural-Networks-and-Optimization)", "학습 갱신"),
    "policy": ("[정책(policy)](Policy)", "정책"),
    "session": ("[한 번의 접속 세션(session)](Terminology-Guide)", "접속 세션"),
    "horizon": ("[미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search)", "미래 탐색 범위"),
    "simulator": ("[환경 시뮬레이터(simulator)](MDP-and-POMDP)", "환경 시뮬레이터"),
    "World": ("[세계(World)](Model-Based-RL-and-World-Models)", "세계"),
    "promotion": ("[다음 난이도로 승급(promotion)](Curriculum-Learning)", "난이도 승급"),
    "region": ("[상태 공간의 영역(region)](Critic-Support-and-OOD)", "영역"),
    "signal": ("[학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)", "학습 신호"),
    "reset": ("[환경 초기화(reset)](Replay-Buffer-and-Episode-Boundaries)", "환경 초기화"),
    "boundary": ("[경계(boundary)](Replay-Buffer-and-Episode-Boundaries)", "경계"),
    "suffix": ("[후속 구간(suffix)](GRU-and-Sequence-Models)", "후속 구간"),
    "leakage": ("[정보 누출(leakage)](Causality-Leakage-and-Evaluation)", "정보 누출"),
    "neural": ("[신경망 기반(neural)](Neural-Networks-and-Optimization)", "신경망 기반"),
    "gradient": ("[기울기(gradient)](Neural-Networks-and-Optimization)", "기울기"),
    "experience": ("[경험(experience)](Replay-Buffer-and-Episode-Boundaries)", "경험"),
    "context": ("[문맥 정보(context)](GRU-and-Sequence-Models)", "문맥 정보"),
    "skill": ("[재사용 가능한 기술(skill)](Skills)", "기술"),
    "ensemble": ("[여러 모델을 함께 쓰는 앙상블(ensemble)](Mixture-Ensemble-and-Calibration)", "앙상블"),
    "identifier": ("[식별자(identifier)](State-Representation)", "식별자"),
    "distance": ("[거리(distance)](Critic-Support-and-OOD)", "거리"),
    "learned": ("[학습된(learned)](Neural-Networks-and-Optimization)", "학습된"),
    "path": ("[경로(path)](Counterfactual-Planning-and-Search)", "경로"),
    "Q-learning": ("[Q-러닝(Q-learning)](Q-Learning-DQN-and-TD)", "Q-러닝"),
    "correctness": ("[의도한 대로 정확히 동작하는지(correctness)](Ablation-Benchmarking-and-Reproducibility)", "정확한 동작 여부"),
    "predicted": ("[예측된(predicted)](Terminology-Guide)", "예측된"),
    "history": ("[기록(history)](Development-History)", "기록"),
    "Full": ("[전체 AASSR 조건(Full)](Experiments)", "전체 AASSR 조건"),
    "effect": ("[효과(effect)](Ablation-Benchmarking-and-Reproducibility)", "효과"),
    "permutation": ("[이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization)", "순열"),
    "beam": ("[유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search)", "빔 탐색"),
    "set": ("[집합(set)](Terminology-Guide)", "집합"),
    "audit": ("[공정성과 구현을 점검하는 감사(audit)](Causality-Leakage-and-Evaluation)", "감사"),
    "tree": ("[탐색 트리(tree)](Counterfactual-Planning-and-Search)", "탐색 트리"),
    "vector": ("[수치 벡터(vector)](Neural-Networks-and-Optimization)", "벡터"),
    "dedup": ("[중복 계산 제거(dedup)](Reproduction)", "중복 제거"),
    "batch": ("[여러 입력 묶음(batch)](Reproduction)", "묶음"),
    "performance": ("[성능(performance)](Ablation-Benchmarking-and-Reproducibility)", "성능"),
    "mass": ("[확률 질량(mass)](Stochasticity-Uncertainty-and-Probability)", "확률 질량"),
    "chance": ("[환경의 확률 분기(chance)](Chance-and-Decision-Nodes)", "환경 확률 분기"),
    "function": ("[함수(function)](Terminology-Guide)", "함수"),
    "model-free": ("[환경 예측 모델 없이 직접 학습하는(model-free)](Reinforcement-Learning)", "환경 예측 모델 없는"),
    "macro": ("[여러 행동을 묶은 상위 행동(macro)](Hierarchical-RL-and-Skills)", "행동 묶음"),
    "guard": ("[잘못된 행동을 제한하는 보호 규칙(guard)](ASEQ)", "보호 규칙"),
    "condition": ("[실험 조건(condition)](Ablation-Benchmarking-and-Reproducibility)", "실험 조건"),
    "compute": ("[계산(compute)](Reproduction)", "계산"),
    "scenario": ("[실험 시나리오(scenario)](Experiments)", "실험 시나리오"),
    "model-based": ("[환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models)", "모델 기반"),
    "channel": ("[정보 채널(channel)](Causality-Leakage-and-Evaluation)", "정보 채널"),
    "factual": ("[실제 사실에 근거한(factual)](Causality-Leakage-and-Evaluation)", "실제 사실 기반"),
    "bias": ("[편향(bias)](Ablation-Benchmarking-and-Reproducibility)", "편향"),
    "recurrent": ("[과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models)", "순환형"),
    "oracle": ("[정답을 알고 있는 기준(oracle)](Ablation-Benchmarking-and-Reproducibility)", "정답을 아는 기준"),
    "Oracle": ("[정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility)", "정답을 아는 기준"),
    "marginal": ("[다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility)", "추가 기여"),
    "random": ("[무작위(random)](Ablation-Benchmarking-and-Reproducibility)", "무작위"),
    "Random": ("[무작위(Random)](Ablation-Benchmarking-and-Reproducibility)", "무작위"),
    "parameter": ("[학습 파라미터(parameter)](Neural-Networks-and-Optimization)", "파라미터"),
    "space": ("[공간(space)](MDP-and-POMDP)", "공간"),
    "IMPORTANT": ("**중요**", "**중요**"),
    "explicit": ("[명시적인(explicit)](Causality-Leakage-and-Evaluation)", "명시적인"),
    "local": ("[현재 주변에 한정된 국소적(local)](Critic-Support-and-OOD)", "국소적"),
    "extrapolation": ("[학습 범위 밖으로 값을 추정하는 외삽(extrapolation)](Critic-Support-and-OOD)", "외삽"),
    "bonus": ("[추가 점수(bonus)](Information-Theory-and-Intrinsic-Motivation)", "추가 점수"),
    "step": ("[단계(step)](Terminology-Guide)", "단계"),
    "variable": ("[변수(variable)](Terminology-Guide)", "변수"),
    "mechanism": ("[작동 원리(mechanism)](Evidence-Matrix)", "작동 원리"),
    "metadata": ("[부가 정보(metadata)](State-Representation)", "부가 정보"),
    "world-model": ("[세계 모델(world-model)](Model-Based-RL-and-World-Models)", "세계 모델"),
    "persistent": ("[에피소드가 끝나도 유지되는(persistent)](Knowledge)", "지속적으로 유지되는"),
    "design": ("[설계(design)](Design-Rationale)", "설계"),
    "counterfactual": ("[실제로 하지 않은 경우를 가정하는 반사실적(counterfactual)](Counterfactual-Planning-and-Search)", "반사실적"),
    "preservation": ("[의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)", "보존"),
    "novelty": ("[새로움(novelty)](Information-Theory-and-Intrinsic-Motivation)", "새로움"),
    "mask": ("[가능/불가능을 표시하는 마스크(mask)](Terminology-Guide)", "마스크"),
    "discounted": ("[미래 보상을 시간에 따라 할인한(discounted)](Value-Functions-and-Bellman-Equation)", "할인된"),
    "probability-weighted": ("[확률로 가중한(probability-weighted)](Chance-and-Decision-Nodes)", "확률 가중"),
    "next-state": ("[다음 상태(next-state)](MDP-and-POMDP)", "다음 상태"),
    "countdown": ("[남은 횟수 카운트다운(countdown)](Causality-Leakage-and-Evaluation)", "남은 횟수"),
    "truth": ("[환경 내부의 실제값(truth)](Causality-Leakage-and-Evaluation)", "환경 내부 실제값"),
    "guided": ("[정답 경로로 유도된(guided)](Causality-Leakage-and-Evaluation)", "정답 경로 유도"),
    "trade-off": ("[한쪽을 얻으면 다른 쪽을 잃는 상충 관계(trade-off)](Terminology-Guide)", "상충 관계"),
    "variance": ("[분산(variance)](Stochasticity-Uncertainty-and-Probability)", "분산"),
    "run": ("[실험 실행(run)](Reproduction)", "실험 실행"),
    "unreliable": ("[신뢰하기 어려운(unreliable)](Calibration)", "신뢰하기 어려운"),
    "knowledge": ("[지식(knowledge)](Knowledge)", "지식"),
    "Concrete": ("[실제 개체를 구분하는(Concrete)](State-Representation)", "실제 개체 구분"),
    "gain": ("[증가량(gain)](Ablation-Benchmarking-and-Reproducibility)", "증가량"),
    "latent": ("[직접 관측되지 않는 잠재 표현(latent)](GRU-and-Sequence-Models)", "잠재 표현"),
    "pruning": ("[유망하지 않은 탐색 가지를 제거하는 가지치기(pruning)](Counterfactual-Planning-and-Search)", "가지치기"),
    "deterministic": ("[같은 입력이면 항상 같은 결과인 결정론적(deterministic)](Stochasticity-Uncertainty-and-Probability)", "결정론적"),
    "noise": ("[잡음(noise)](Stochasticity-Uncertainty-and-Probability)", "잡음"),
    "residual": ("[기본 값에 더하는 잔차(residual)](Policy)", "잔차"),
    "research": ("[연구(research)](Research-Questions)", "연구"),
    "repetition": ("[반복(repetition)](ASEQ)", "반복"),
    "adapter": ("[서로 다른 입력·행동 형식을 연결하는 변환기(adapter)](Experiments)", "변환 어댑터"),
    "test": ("[검사 또는 테스트(test)](Ablation-Benchmarking-and-Reproducibility)", "테스트"),
    "switch": ("[행동 전환(switch)](Imagination)", "행동 전환"),
    "approximation": ("[근사(approximation)](Value-Functions-and-Bellman-Equation)", "근사"),
    "next": ("[다음(next)](Terminology-Guide)", "다음"),
    "encoding": ("[학습용 수치 표현으로 바꾸는 인코딩(encoding)](State-Representation)", "인코딩"),
    "suppression": ("[후보 억제(suppression)](ASEQ)", "후보 억제"),
    "execution": ("[실제 실행(execution)](Research-Jargon-Guide)", "실행"),
    "belief": ("[관측을 바탕으로 추정한 상태 믿음(belief)](MDP-and-POMDP)", "상태 추정"),
    "conditional": ("[조건부(conditional)](Stochasticity-Uncertainty-and-Probability)", "조건부"),
    "higher-level": ("[여러 기본 행동을 묶는 상위 수준(higher-level)](Hierarchical-RL-and-Skills)", "상위 수준"),
    "weight": ("[가중치(weight)](Neural-Networks-and-Optimization)", "가중치"),
    "collapse": ("[여러 결과가 하나로 뭉개지는 붕괴(collapse)](Mixture-Ensemble-and-Calibration)", "붕괴"),
    "global": ("[전체 범위(global)](Terminology-Guide)", "전체 범위"),
    "intermediate": ("[중간(intermediate)](Sparse-Reward-and-Credit-Assignment)", "중간"),
    "primary": ("[주요(primary)](Research-Questions)", "주요"),
    "randomness": ("[무작위성(randomness)](Stochasticity-Uncertainty-and-Probability)", "무작위성"),
    "shortcut": ("[정답 정보를 우회적으로 이용하는 지름길(shortcut)](Causality-Leakage-and-Evaluation)", "지름길"),
    "time": ("[시간(time)](Terminology-Guide)", "시간"),
    "scalar": ("[숫자 하나인 스칼라(scalar)](Neural-Networks-and-Optimization)", "스칼라"),
    "pass": ("[검사를 통과(pass)](Ablation-Benchmarking-and-Reproducibility)", "통과"),
    "empirical": ("[실제 관측 경험에 근거한(empirical)](Ablation-Benchmarking-and-Reproducibility)", "경험적"),
    "goal": ("[최종 목표(goal)](Sparse-Reward-Problem)", "목표"),
    "hindsight": ("[결과를 본 뒤 얻은 사후 정보(hindsight)](Causality-Leakage-and-Evaluation)", "사후 정보"),
    "lockout": ("[복구할 수 없는 실패 잠금(lockout)](Replay-Buffer-and-Episode-Boundaries)", "실패 잠금"),
    "pressure": ("[환경 내부의 숨은 압박 값(pressure)](Causality-Leakage-and-Evaluation)", "숨은 압박 값"),
    "control": ("[효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility)", "대조 조건"),
    "node": ("[탐색 트리의 한 지점(node)](Chance-and-Decision-Nodes)", "노드"),
    "expected": ("[확률을 고려해 기대되는(expected)](Chance-and-Decision-Nodes)", "기대되는"),
    "limit": ("[제한(limit)](Terminology-Guide)", "제한"),
    "curiosity": ("[새 정보를 찾아보려는 호기심 기반 탐색(curiosity)](Information-Theory-and-Intrinsic-Motivation)", "호기심 기반 탐색"),
    "discovery": ("[스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions)", "발견"),
    "advantage": ("[다른 선택보다 나은 정도(advantage)](Value-Functions-and-Bellman-Equation)", "상대적 이점"),
    "search": ("[탐색(search)](Counterfactual-Planning-and-Search)", "탐색"),
    "dataset": ("[데이터 묶음(dataset)](Ablation-Benchmarking-and-Reproducibility)", "데이터셋"),
    "rule": ("[규칙(rule)](Terminology-Guide)", "규칙"),
    "autonomous": ("[사람의 정답 경로 없이 자율적인(autonomous)](Research-Questions)", "자율적인"),
    "behavior": ("[행동 양상(behavior)](Experiments)", "행동 양상"),
    "similarity": ("[유사도(similarity)](Critic-Support-and-OOD)", "유사도"),
    "reached": ("[도달한(reached)](Curriculum-Learning)", "도달한"),
    "focused": ("[특정 범위에 집중한(focused)](Experiments)", "집중형"),
    "experiment": ("[실험(experiment)](Experiments)", "실험"),
    "key": ("[핵심(key)](Terminology-Guide)", "핵심"),
    "recall": ("[놓치지 않고 찾아낸 비율인 재현율(recall)](Ablation-Benchmarking-and-Reproducibility)", "재현율"),
    "optimization": ("[최적화(optimization)](Neural-Networks-and-Optimization)", "최적화"),
    "online": ("[경험이 들어올 때마다 갱신하는 온라인 방식(online)](Neural-Networks-and-Optimization)", "온라인 방식"),
    "progress": ("[진행도(progress)](Terminology-Guide)", "진행도"),
    "continuation": ("[계속 진행되는 상태(continuation)](Chance-and-Decision-Nodes)", "계속 진행"),
    "factor": ("[실험에서 바꾸어 보는 요인(factor)](Ablation-Benchmarking-and-Reproducibility)", "실험 요인"),
    "temporal": ("[시간 순서를 고려하는(temporal)](GRU-and-Sequence-Models)", "시간 순서 기반"),
    "entropy": ("[확률 분포의 불확실성을 나타내는 엔트로피(entropy)](Information-Theory-and-Intrinsic-Motivation)", "엔트로피"),
    "imbalance": ("[데이터 수의 불균형(imbalance)](Loss-Functions-and-Class-Imbalance)", "불균형"),
    "option": ("[여러 기본 행동을 묶은 상위 행동 단위(option)](Hierarchical-RL-and-Skills)", "상위 행동 단위"),
    "alias": ("[같은 구조를 가리키는 다른 이름(alias)](State-Representation)", "별칭"),
    "repaired": ("[문제를 수정한 뒤의(repaired)](Development-History)", "수정 후"),
    "repair": ("[문제 수정(repair)](Development-History)", "문제 수정"),
    "higher": ("[더 높은 단계(higher)](Curriculum-Learning)", "더 높은"),
    "trace": ("[과정을 추적한 기록(trace)](Development-History)", "추적 기록"),
    "unavailable": ("[현재 사용할 수 없는(unavailable)](Terminology-Guide)", "현재 사용 불가"),
    "commit": ("[Git 변경 기록 단위(commit)](Research-Jargon-Guide)", "커밋"),
    "category": ("[범주(category)](Loss-Functions-and-Class-Imbalance)", "범주"),
    "greedy": ("[현재 추정값이 가장 큰 행동만 고르는 탐욕 선택(greedy)](Exploration-and-Exploitation)", "탐욕 선택"),
    "legacy": ("[구버전 호환 코드(legacy)](Development-History)", "구버전"),
    "mismatch": ("[서로 맞지 않는 불일치(mismatch)](Causality-Leakage-and-Evaluation)", "불일치"),
    "schedule": ("[학습 진행 스케줄(schedule)](Curriculum-Learning)", "학습 스케줄"),
    "memorization": ("[이름이나 사례를 그대로 외우는 암기(memorization)](Relational-Representation-and-Generalization)", "암기"),
    "legal-mask": ("[가능 행동 마스크(legal-mask)](Prophecy)", "가능 행동 마스크"),
    "plan": ("[계획(plan)](Counterfactual-Planning-and-Search)", "계획"),
    "direct": ("[직접적인(direct)](Terminology-Guide)", "직접적인"),
    "success-producing": ("[실제로 성공을 만들어내는(success-producing)](Experiments)", "성공을 만들어내는"),
    "calls": ("[모델 호출 횟수(calls)](Reproduction)", "호출 횟수"),
    "aggregate": ("[여러 결과를 합친 종합값(aggregate)](Ablation-Benchmarking-and-Reproducibility)", "종합값"),
    "difficulty": ("[난이도(difficulty)](Curriculum-Learning)", "난이도"),
    "official": ("[공식 구현(official)](Experiments)", "공식"),
    "label": ("[정답 범주 표시(label)](Loss-Functions-and-Class-Imbalance)", "라벨"),
    "response-causal": ("[실제 응답에서 원인 순서를 지키는(response-causal)](Causality-Leakage-and-Evaluation)", "응답 인과성 보장"),
    "known": ("[이미 알려진(known)](Terminology-Guide)", "알려진"),
    "branching": ("[여러 미래로 갈라지는 분기(branching)](Chance-and-Decision-Nodes)", "분기"),
    "Terminal": ("[에피소드 종료(Terminal)](Replay-Buffer-and-Episode-Boundaries)", "에피소드 종료"),
    "bind": ("[역할을 실제 객체에 연결(bind)](Skills)", "연결"),
    "discount": ("[미래 보상의 할인율(discount)](Value-Functions-and-Bellman-Equation)", "할인율"),
    "fixed": ("[고정된(fixed)](Ablation-Benchmarking-and-Reproducibility)", "고정된"),
    "optimizer": ("[신경망 파라미터를 갱신하는 최적화 알고리즘(optimizer)](Neural-Networks-and-Optimization)", "최적화 알고리즘"),
    "normalization": ("[수치 범위를 맞추는 정규화(normalization)](Neural-Networks-and-Optimization)", "정규화"),
    "invariance": ("[이름 등이 바뀌어도 결과가 유지되는 불변성(invariance)](Relational-Representation-and-Generalization)", "불변성"),
    "zero-memory": ("[과거 기억을 0으로 초기화한(zero-memory)](GRU-and-Sequence-Models)", "기억 0 초기화"),
    "inference": ("[학습된 모델로 값을 계산하는 추론(inference)](Neural-Networks-and-Optimization)", "추론"),
    "usage": ("[사용량(usage)](Terminology-Guide)", "사용량"),
    "leak": ("[정보 누출(leak)](Causality-Leakage-and-Evaluation)", "정보 누출"),
    "point": ("[지점(point)](Terminology-Guide)", "지점"),
    "query": ("[조회 또는 질의(query)](Terminology-Guide)", "질의"),
    "real-training": ("[실제 환경 경험으로 학습한(real-training)](Critic-Support-and-OOD)", "실제 환경 학습"),
    "observed": ("[실제로 관측된(observed)](Causality-Leakage-and-Evaluation)", "관측된"),
    "milestone": ("[학습 진행의 도달 기준점(milestone)](Curriculum-Learning)", "도달 기준점"),
    "reach": ("[도달(reach)](Curriculum-Learning)", "도달"),
    "unsupported": ("[실제 데이터 근거가 부족한(unsupported)](Critic-Support-and-OOD)", "근거 부족"),
    "rebinding": ("[새 문제의 실제 객체에 다시 연결하는 것(rebinding)](Skills)", "새 객체 재연결"),
}

PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(re.escape(t) for t in sorted(TERMS, key=lambda x: (-len(x), x)))
    + r")(?![A-Za-z0-9_-])"
)


def transform_plain(text: str, seen: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        term = match.group(0)
        first, later = TERMS[term]
        if term not in seen:
            seen.add(term)
            return first
        return later
    return TERM_RE.sub(repl, text)


def transform_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    out: list[str] = []
    in_fence = False
    seen: set[str] = set()
    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or stripped.startswith("#"):
            out.append(line)
            continue
        parts = PROTECTED.split(line)
        out.append("".join(
            part if i % 2 else transform_plain(part, seen)
            for i, part in enumerate(parts)
        ))
    updated = "".join(out)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    changed = []
    for path in sorted(WIKI.glob("*.md")):
        if path.name == "README.md":
            continue
        if transform_file(path):
            changed.append(path.as_posix())
    print(f"High-frequency jargon pass changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
