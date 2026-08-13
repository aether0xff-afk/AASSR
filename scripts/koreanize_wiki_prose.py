from __future__ import annotations

import re
from pathlib import Path

WIKI = Path("wiki")

# This pass is intentionally conservative:
# - fenced code blocks are untouched
# - inline code is untouched
# - existing Markdown links/images are untouched
# - each general technical term is introduced once as Korean(English)+link,
#   then later occurrences become Korean
# - AASSR module/proper names remain clickable every time

TERMS: dict[str, tuple[str, str]] = {
    "current architecture": ("[현재 구조(current architecture)](Current-Status)", "현재 구조"),
    "source of truth": ("[최종 기준(source of truth)](Current-Status)", "최종 기준"),
    "same-checkpoint": ("[같은 체크포인트(same-checkpoint)](Experiments)", "같은 체크포인트"),
    "final blind": ("[최종 비공개 평가(final blind)](Ablation-Benchmarking-and-Reproducibility)", "최종 비공개 평가"),
    "counterfactual planning": ("[반사실적 계획(counterfactual planning)](Counterfactual-Planning-and-Search)", "반사실적 계획"),
    "Counterfactual Planning": ("[반사실적 계획(Counterfactual Planning)](Counterfactual-Planning-and-Search)", "반사실적 계획"),
    "partial observability": ("[부분 관측(partial observability)](MDP-and-POMDP)", "부분 관측"),
    "Partial Observability": ("[부분 관측(Partial Observability)](MDP-and-POMDP)", "부분 관측"),
    "sparse reward": ("[희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)", "희소 보상"),
    "Sparse Reward": ("[희소 보상(Sparse Reward)](Sparse-Reward-and-Credit-Assignment)", "희소 보상"),
    "dense reward": ("[밀집 보상(dense reward)](Sparse-Reward-and-Credit-Assignment)", "밀집 보상"),
    "Dense Reward": ("[밀집 보상(Dense Reward)](Sparse-Reward-and-Credit-Assignment)", "밀집 보상"),
    "credit assignment": ("[보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment)", "보상 책임 배분"),
    "Credit Assignment": ("[보상 책임 배분(Credit Assignment)](Sparse-Reward-and-Credit-Assignment)", "보상 책임 배분"),
    "reinforcement learning": ("[강화학습(reinforcement learning)](Reinforcement-Learning)", "강화학습"),
    "Reinforcement Learning": ("[강화학습(Reinforcement Learning)](Reinforcement-Learning)", "강화학습"),
    "relational representation": ("[관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)", "관계 기반 표현"),
    "Relational Representation": ("[관계 기반 표현(Relational Representation)](Relational-Representation-and-Generalization)", "관계 기반 표현"),
    "semantic state": ("[의미 기반 상태(semantic state)](State-Representation)", "의미 기반 상태"),
    "concrete identity": ("[실제 개체 구분(concrete identity)](State-Representation)", "실제 개체 구분"),
    "concrete action": ("[실제 실행 행동(concrete action)](State-Representation)", "실제 실행 행동"),
    "world model": ("[세계 모델(world model)](Model-Based-RL-and-World-Models)", "세계 모델"),
    "World Model": ("[세계 모델(World Model)](Model-Based-RL-and-World-Models)", "세계 모델"),
    "model-based RL": ("[모델 기반 강화학습(model-based RL)](Model-Based-RL-and-World-Models)", "모델 기반 강화학습"),
    "Model-based RL": ("[모델 기반 강화학습(Model-based RL)](Model-Based-RL-and-World-Models)", "모델 기반 강화학습"),
    "model-free RL": ("[환경 예측 모델을 직접 쓰지 않는 강화학습(model-free RL)](Reinforcement-Learning)", "환경 예측 모델을 직접 쓰지 않는 강화학습"),
    "outcome probability": ("[결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability)", "결과 확률"),
    "Outcome probability": ("[결과 확률(Outcome probability)](Stochasticity-Uncertainty-and-Probability)", "결과 확률"),
    "prediction reliability": ("[예측 신뢰도(prediction reliability)](Calibration)", "예측 신뢰도"),
    "Prediction reliability": ("[예측 신뢰도(Prediction reliability)](Calibration)", "예측 신뢰도"),
    "probability mass": ("[확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability)", "확률 질량"),
    "legal action mask": ("[가능 행동 마스크(legal action mask)](Prophecy)", "가능 행동 마스크"),
    "legal-action mask": ("[가능 행동 마스크(legal-action mask)](Prophecy)", "가능 행동 마스크"),
    "chance node": ("[환경 결과 노드(chance node)](Chance-and-Decision-Nodes)", "환경 결과 노드"),
    "Chance node": ("[환경 결과 노드(Chance node)](Chance-and-Decision-Nodes)", "환경 결과 노드"),
    "decision node": ("[행동 선택 노드(decision node)](Chance-and-Decision-Nodes)", "행동 선택 노드"),
    "Decision node": ("[행동 선택 노드(Decision node)](Chance-and-Decision-Nodes)", "행동 선택 노드"),
    "local support": ("[국소 데이터 근거(local support)](Critic-Support-and-OOD)", "국소 데이터 근거"),
    "Local support": ("[국소 데이터 근거(Local support)](Critic-Support-and-OOD)", "국소 데이터 근거"),
    "Critic support": ("[Critic 데이터 근거(Critic support)](Critic-Support-and-OOD)", "Critic 데이터 근거"),
    "Critic Support": ("[Critic 데이터 근거(Critic Support)](Critic-Support-and-OOD)", "Critic 데이터 근거"),
    "self-loop": ("[제자리 반복(self-loop)](ASEQ)", "제자리 반복"),
    "Self-loop": ("[제자리 반복(Self-loop)](ASEQ)", "제자리 반복"),
    "replay buffer": ("[경험 저장소(replay buffer)](Replay-Buffer-and-Episode-Boundaries)", "경험 저장소"),
    "Replay Buffer": ("[경험 저장소(Replay Buffer)](Replay-Buffer-and-Episode-Boundaries)", "경험 저장소"),
    "intrinsic motivation": ("[내재 동기(intrinsic motivation)](Information-Theory-and-Intrinsic-Motivation)", "내재 동기"),
    "Intrinsic Motivation": ("[내재 동기(Intrinsic Motivation)](Information-Theory-and-Intrinsic-Motivation)", "내재 동기"),
    "epistemic uncertainty": ("[지식 부족에서 오는 불확실성(epistemic uncertainty)](Stochasticity-Uncertainty-and-Probability)", "지식 부족에서 오는 불확실성"),
    "Epistemic Uncertainty": ("[지식 부족에서 오는 불확실성(Epistemic Uncertainty)](Stochasticity-Uncertainty-and-Probability)", "지식 부족에서 오는 불확실성"),
    "distribution shift": ("[데이터 분포 변화(distribution shift)](Critic-Support-and-OOD)", "데이터 분포 변화"),
    "Distribution Shift": ("[데이터 분포 변화(Distribution Shift)](Critic-Support-and-OOD)", "데이터 분포 변화"),
    "model exploitation": ("[모델 오류 악용(model exploitation)](Model-Based-RL-and-World-Models)", "모델 오류 악용"),
    "Model Exploitation": ("[모델 오류 악용(Model Exploitation)](Model-Based-RL-and-World-Models)", "모델 오류 악용"),
    "status-aware": ("[상태 코드까지 고려하는(status-aware)](Calibration)", "상태 코드까지 고려하는"),
    "status-balanced": ("[상태 코드 데이터 불균형을 보정한(status-balanced)](Prophecy)", "상태 코드 데이터 불균형을 보정한"),
    "conditional-mixture": ("[조건부 혼합(conditional-mixture)](Prophecy)", "조건부 혼합"),
    "information residual": ("[정보 가치 잔차(information residual)](Policy)", "정보 가치 잔차"),
    "Information residual": ("[정보 가치 잔차(Information residual)](Policy)", "정보 가치 잔차"),
    "information-value residual": ("[정보 가치 잔차(information-value residual)](Policy)", "정보 가치 잔차"),
    "Q-value": ("[Q값(Q-value)](Value-Functions-and-Bellman-Equation)", "Q값"),
    "Q value": ("[Q값(Q value)](Value-Functions-and-Bellman-Equation)", "Q값"),
    "final evaluation": ("[최종 평가(final evaluation)](Ablation-Benchmarking-and-Reproducibility)", "최종 평가"),
    "regression test": ("[회귀 테스트(regression test)](Ablation-Benchmarking-and-Reproducibility)", "회귀 테스트"),
    "regression": ("[회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)", "회귀 검증"),
    "benchmark": ("[표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)", "표준 비교 실험"),
    "Benchmark": ("[표준 비교 실험(Benchmark)](Ablation-Benchmarking-and-Reproducibility)", "표준 비교 실험"),
    "baseline": ("[비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)", "비교 기준"),
    "Baseline": ("[비교 기준(Baseline)](Ablation-Benchmarking-and-Reproducibility)", "비교 기준"),
    "ablation": ("[구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility)", "구성요소 제거 비교"),
    "Ablation": ("[구성요소 제거 비교(Ablation)](Ablation-Benchmarking-and-Reproducibility)", "구성요소 제거 비교"),
    "metric": ("[평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)", "평가지표"),
    "metrics": ("[평가지표(metrics)](Ablation-Benchmarking-and-Reproducibility)", "평가지표"),
    "unseen": ("[학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization)", "학습 중 보지 못한"),
    "transfer": ("[전이(transfer)](Relational-Representation-and-Generalization)", "전이"),
    "generalization": ("[일반화(generalization)](Relational-Representation-and-Generalization)", "일반화"),
    "Generalization": ("[일반화(Generalization)](Relational-Representation-and-Generalization)", "일반화"),
    "truncation": ("[외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)", "외부 제한 종료"),
    "terminal": ("[에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries)", "에피소드 종료"),
    "bootstrap": ("[다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)", "다음 상태 가치 이어받기"),
    "curriculum": ("[난이도 조절 학습(curriculum)](Curriculum-Learning)", "난이도 조절 학습"),
    "Curriculum": ("[난이도 조절 학습(Curriculum)](Curriculum-Learning)", "난이도 조절 학습"),
    "exploration": ("[탐색(exploration)](Exploration-and-Exploitation)", "탐색"),
    "Exploration": ("[탐색(Exploration)](Exploration-and-Exploitation)", "탐색"),
    "exploitation": ("[활용(exploitation)](Exploration-and-Exploitation)", "활용"),
    "Exploitation": ("[활용(Exploitation)](Exploration-and-Exploitation)", "활용"),
    "holdout": ("[검증용 분리 데이터(holdout)](Calibration)", "검증용 분리 데이터"),
    "Holdout": ("[검증용 분리 데이터(Holdout)](Calibration)", "검증용 분리 데이터"),
    "checkpoint": ("[체크포인트(checkpoint)](Reproduction)", "체크포인트"),
    "seed": ("[난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)", "난수 시드"),
    "intervention": ("[실제 행동 개입(intervention)](Imagination)", "실제 행동 개입"),
    "Intervention": ("[실제 행동 개입(Intervention)](Imagination)", "실제 행동 개입"),
    "wall-clock": ("[실제 실행 시간(wall-clock)](Reproduction)", "실제 실행 시간"),
    "throughput": ("[처리량(throughput)](Reproduction)", "처리량"),
    "batching": ("[묶음 처리(batching)](Reproduction)", "묶음 처리"),
    "loss": ("[학습 손실(loss)](Loss-Functions-and-Class-Imbalance)", "학습 손실"),
    "Loss": ("[학습 손실(Loss)](Loss-Functions-and-Class-Imbalance)", "학습 손실"),
    "agent": ("[에이전트(agent)](Reinforcement-Learning)", "에이전트"),
    "Agent": ("[에이전트(Agent)](Reinforcement-Learning)", "에이전트"),
    "environment": ("[환경(environment)](Reinforcement-Learning)", "환경"),
    "Environment": ("[환경(Environment)](Reinforcement-Learning)", "환경"),
    "observation": ("[관측(observation)](MDP-and-POMDP)", "관측"),
    "Observation": ("[관측(Observation)](MDP-and-POMDP)", "관측"),
    "representation": ("[표현(representation)](Relational-Representation-and-Generalization)", "표현"),
    "Representation": ("[표현(Representation)](Relational-Representation-and-Generalization)", "표현"),
    "transition": ("[상태 전이(transition)](MDP-and-POMDP)", "상태 전이"),
    "Transition": ("[상태 전이(Transition)](MDP-and-POMDP)", "상태 전이"),
    "action": ("[행동(action)](Reinforcement-Learning)", "행동"),
    "Action": ("[행동(Action)](Reinforcement-Learning)", "행동"),
    "reward": ("[보상(reward)](Sparse-Reward-and-Credit-Assignment)", "보상"),
    "Reward": ("[보상(Reward)](Sparse-Reward-and-Credit-Assignment)", "보상"),
    "return": ("[누적 보상(return)](Value-Functions-and-Bellman-Equation)", "누적 보상"),
    "Return": ("[누적 보상(Return)](Value-Functions-and-Bellman-Equation)", "누적 보상"),
    "Policy": ("[Policy(정책 모델)](Policy)", "[Policy](Policy)"),
    "Knowledge": ("[Knowledge(에피소드 지식)](Knowledge)", "[Knowledge](Knowledge)"),
    "Prophecy": ("[Prophecy(미래 예측 모델)](Prophecy)", "[Prophecy](Prophecy)"),
    "Calibration": ("[Calibration(예측 신뢰도 보정)](Calibration)", "[Calibration](Calibration)"),
    "Critic": ("[Critic(미래 가치 평가기)](Critic)", "[Critic](Critic)"),
    "Imagination": ("[Imagination(가상 미래 탐색)](Imagination)", "[Imagination](Imagination)"),
    "Skills": ("[Skills(성공 절차 재사용)](Skills)", "[Skills](Skills)"),
    "Skill": ("[Skill(성공 절차 재사용)](Skills)", "[Skill](Skills)"),
    "ASEQ": ("[ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)", "[ASEQ](ASEQ)"),
    "DQN": ("[DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)", "[DQN](Q-Learning-DQN-and-TD)"),
    "GRU": ("[GRU(게이트 순환 유닛)](GRU-and-Sequence-Models)", "[GRU](GRU-and-Sequence-Models)"),
    "OOD": ("[학습 분포 밖(OOD)](Critic-Support-and-OOD)", "[OOD](Critic-Support-and-OOD)"),
    "DreamerV3": ("[DreamerV3(외부 세계 모델 강화학습 비교군)](Experiments)", "[DreamerV3](Experiments)"),
}

# Exact level-1 title translations. File names stay stable so Wiki links do not break.
TITLE_MAP = {
    "# AASSR Wiki": "# AASSR 위키",
    "# AASSR in 5 Minutes": "# AASSR 5분 설명",
    "# Glossary": "# 용어 사전 (Glossary)",
    "# Current Status": "# 현재 연구 상태 (Current Status)",
    "# Research Questions": "# 연구 질문 (Research Questions)",
    "# Experiments": "# 실험 설계와 결과 (Experiments)",
    "# Reproduction": "# 실험 재현 방법 (Reproduction)",
    "# Evidence Matrix": "# 연구 질문-증거 연결표 (Evidence Matrix)",
    "# Research Architecture": "# 연구 구조 (Research Architecture)",
    "# Design Rationale": "# 설계 이유 (Design Rationale)",
    "# State Representation": "# 상태 표현 (State Representation)",
    "# Policy": "# Policy — 기본 정책 모델",
    "# Knowledge": "# Knowledge — 에피소드 지식",
    "# Prophecy": "# Prophecy — 미래 예측 모델",
    "# Calibration": "# Calibration — 예측 신뢰도 보정",
    "# Critic": "# Critic — 미래 가치 평가기",
    "# Imagination": "# Imagination — 가상 미래 탐색",
    "# Skills": "# Skills — 성공 절차 재사용",
    "# Reinforcement Learning": "# 강화학습 (Reinforcement Learning)",
}

PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
TERMS_RE = re.compile(
    "|".join(
        re.escape(term)
        for term in sorted(TERMS, key=lambda x: (-len(x), x))
    )
)


def transform_plain(text: str, seen: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        term = match.group(0)
        first, later = TERMS[term]
        if term not in seen:
            seen.add(term)
            return first
        return later

    return TERMS_RE.sub(repl, text)


def transform_line(line: str, seen: set[str]) -> str:
    if line.rstrip("\n") in TITLE_MAP:
        suffix = "\n" if line.endswith("\n") else ""
        return TITLE_MAP[line.rstrip("\n")] + suffix

    # Keep other headings stable; headings are navigation anchors and excessive
    # auto-linking inside them makes anchors unpredictable.
    if line.lstrip().startswith("#"):
        return line

    parts = PROTECTED.split(line)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        else:
            out.append(transform_plain(part, seen))
    return "".join(out)


def transform_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    seen: set[str] = set()

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        out.append(transform_line(line, seen))

    updated = "".join(out)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    changed: list[str] = []
    for path in sorted(WIKI.glob("*.md")):
        if path.name == "README.md":
            continue
        if transform_file(path):
            changed.append(path.as_posix())

    print(f"Korean-first terminology pass changed {len(changed)} files")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()
