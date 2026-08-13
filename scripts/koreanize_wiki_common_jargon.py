from __future__ import annotations

import re
from pathlib import Path

WIKI = Path("wiki")

# Common research/developer English that is easy for specialists to gloss over,
# but should not be assumed knowledge in a Wiki for amateur developers/researchers.
# Code blocks, inline code, links, and headings are protected.
COMMON_TERMS: dict[str, tuple[str, str]] = {
    "current-generation": ("[현재 세대(current-generation)](Current-Status)", "현재 세대"),
    "current generation": ("[현재 세대(current generation)](Current-Status)", "현재 세대"),
    "Current generation": ("[현재 세대(Current generation)](Current-Status)", "현재 세대"),
    "current runtime": ("[현재 실행 구조(current runtime)](Current-Status)", "현재 실행 구조"),
    "Current runtime": ("[현재 실행 구조(Current runtime)](Current-Status)", "현재 실행 구조"),
    "runtime contract": ("[실행 구조 명세(runtime contract)](Current-Status)", "실행 구조 명세"),
    "architecture contract": ("[구조 명세(architecture contract)](Current-Status)", "구조 명세"),
    "component contract": ("[구성요소 명세(component contract)](Current-Status)", "구성요소 명세"),
    "research seed": ("[연구용 난수 시드(research seed)](Ablation-Benchmarking-and-Reproducibility)", "연구용 난수 시드"),
    "final evidence": ("[최종 증거(final evidence)](Evidence-Matrix)", "최종 증거"),
    "performance evidence": ("[성능 증거(performance evidence)](Evidence-Matrix)", "성능 증거"),
    "mechanism evidence": ("[메커니즘 증거(mechanism evidence)](Evidence-Matrix)", "메커니즘 증거"),
    "proxy metric": ("[대리 평가지표(proxy metric)](Ablation-Benchmarking-and-Reproducibility)", "대리 평가지표"),
    "final metric": ("[최종 평가지표(final metric)](Ablation-Benchmarking-and-Reproducibility)", "최종 평가지표"),
    "reward shaping": ("[중간 보상 설계(reward shaping)](Sparse-Reward-and-Credit-Assignment)", "중간 보상 설계"),
    "action surface": ("[현재 가능한 행동 집합(action surface)](Terminology-Guide)", "현재 가능한 행동 집합"),
    "legal action": ("[현재 허용된 행동(legal action)](Terminology-Guide)", "현재 허용된 행동"),
    "public state": ("[공개 관측 상태(public state)](State-Representation)", "공개 관측 상태"),
    "hidden state": ("[숨은 환경 상태(hidden state)](MDP-and-POMDP)", "숨은 환경 상태"),
    "true state": ("[실제 환경 상태(true state)](MDP-and-POMDP)", "실제 환경 상태"),
    "real transition": ("[실제 상태 전이(real transition)](Causality-Leakage-and-Evaluation)", "실제 상태 전이"),
    "imagined transition": ("[가상 상태 전이(imagined transition)](Causality-Leakage-and-Evaluation)", "가상 상태 전이"),
    "real data": ("[실제 데이터(real data)](Causality-Leakage-and-Evaluation)", "실제 데이터"),
    "training data": ("[학습 데이터(training data)](Terminology-Guide)", "학습 데이터"),
    "validation data": ("[검증 데이터(validation data)](Ablation-Benchmarking-and-Reproducibility)", "검증 데이터"),
    "decision-critical": ("[의사결정에 중요한(decision-critical)](Calibration)", "의사결정에 중요한"),
    "fail-closed": ("[근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD)", "근거가 부족하면 보수적으로 거부하는"),
    "fail closed": ("[근거가 부족하면 보수적으로 거부하는(fail closed)](Critic-Support-and-OOD)", "근거가 부족하면 보수적으로 거부하는"),
    "root action": ("[지금 실제로 실행할 첫 행동(root action)](Imagination)", "지금 실제로 실행할 첫 행동"),
    "root candidate": ("[첫 행동 후보(root candidate)](Imagination)", "첫 행동 후보"),
    "root": ("[탐색의 첫 행동(root)](Imagination)", "탐색의 첫 행동"),
    "override": ("[기본 행동 덮어쓰기(override)](Imagination)", "기본 행동 덮어쓰기"),
    "fallback": ("[기본 경로로 돌아가기(fallback)](Imagination)", "기본 경로로 돌아가기"),
    "margin": ("[최소 차이 기준(margin)](Imagination)", "최소 차이 기준"),
    "gate": ("[판정 관문(gate)](Terminology-Guide)", "판정 관문"),
    "gating": ("[조건부 통과 판단(gating)](Terminology-Guide)", "조건부 통과 판단"),
    "planner": ("[계획기(planner)](Counterfactual-Planning-and-Search)", "계획기"),
    "planning": ("[계획(planning)](Counterfactual-Planning-and-Search)", "계획"),
    "prediction": ("[예측(prediction)](Terminology-Guide)", "예측"),
    "predictor": ("[예측 모델(predictor)](Terminology-Guide)", "예측 모델"),
    "dynamics": ("[환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)", "환경의 상태 변화 규칙"),
    "learner": ("[학습 주체(learner)](Terminology-Guide)", "학습 주체"),
    "training": ("[학습(training)](Terminology-Guide)", "학습"),
    "validation": ("[검증(validation)](Ablation-Benchmarking-and-Reproducibility)", "검증"),
    "evaluation": ("[평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)", "평가"),
    "diagnostic": ("[진단 실험(diagnostic)](Evidence-Matrix)", "진단 실험"),
    "evidence": ("[증거(evidence)](Evidence-Matrix)", "증거"),
    "claim": ("[연구 주장(claim)](Evidence-Matrix)", "연구 주장"),
    "claims": ("[연구 주장(claims)](Evidence-Matrix)", "연구 주장"),
    "architecture": ("[구조(architecture)](Research-Architecture)", "구조"),
    "runtime": ("[실행 구조(runtime)](Current-Status)", "실행 구조"),
    "contract": ("[명세(contract)](Current-Status)", "명세"),
    "component": ("[구성요소(component)](Research-Architecture)", "구성요소"),
    "active": ("[현재 활성(active)](Current-Status)", "현재 활성"),
    "historical": ("[과거 기록(historical)](Development-History)", "과거 기록"),
    "current": ("[현재(current)](Current-Status)", "현재"),
    "public": ("[공개된(public)](State-Representation)", "공개된"),
    "hidden": ("[숨겨진(hidden)](MDP-and-POMDP)", "숨겨진"),
    "response": ("[응답(response)](State-Representation)", "응답"),
    "status": ("[상태 코드(status)](Terminology-Guide)", "상태 코드"),
    "objective": ("[학습 목표(objective)](Terminology-Guide)", "학습 목표"),
    "categorical": ("[범주형(categorical)](Loss-Functions-and-Class-Imbalance)", "범주형"),
    "rare": ("[드문(rare)](Loss-Functions-and-Class-Imbalance)", "드문"),
    "reliability": ("[신뢰도(reliability)](Calibration)", "신뢰도"),
    "stochastic": ("[확률적(stochastic)](Stochasticity-Uncertainty-and-Probability)", "확률적"),
    "relational": ("[관계 기반(relational)](Relational-Representation-and-Generalization)", "관계 기반"),
    "multimodal": ("[여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration)", "여러 결과 형태를 가진"),
    "episode-local": ("[현재 에피소드 안에서만 유지되는(episode-local)](Knowledge)", "현재 에피소드 안에서만 유지되는"),
    "episode": ("[한 번의 문제 풀이 구간(episode)](Terminology-Guide)", "한 번의 문제 풀이 구간"),
    "provenance": ("[정보의 출처 기록(provenance)](Knowledge)", "정보의 출처 기록"),
    "value": ("[가치(value)](Value-Functions-and-Bellman-Equation)", "가치"),
    "model": ("[학습 모델(model)](Terminology-Guide)", "학습 모델"),
    "network": ("[신경망(network)](Neural-Networks-and-Optimization)", "신경망"),
    "input": ("[입력(input)](Terminology-Guide)", "입력"),
    "output": ("[출력(output)](Terminology-Guide)", "출력"),
    "identity": ("[식별 방식(identity)](State-Representation)", "식별 방식"),
    "layer": ("[처리 계층(layer)](Research-Architecture)", "처리 계층"),
    "framework": ("[문제 표현 틀(framework)](Terminology-Guide)", "문제 표현 틀"),
    "success": ("[성공(success)](Terminology-Guide)", "성공"),
    "failure": ("[실패(failure)](Replay-Buffer-and-Episode-Boundaries)", "실패"),
    "support": ("[데이터 근거(support)](Critic-Support-and-OOD)", "데이터 근거"),
    "protocol": ("[실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)", "실험 규칙"),
    "frozen": ("[학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility)", "학습을 멈춘"),
    "proxy": ("[대리 지표(proxy)](Ablation-Benchmarking-and-Reproducibility)", "대리 지표"),
    "state": ("[상태(state)](State-Representation)", "상태"),
    "State": ("[상태(State)](State-Representation)", "상태"),
}

PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")

# Longest terms first and explicit token boundaries prevent accidental changes
# inside identifiers such as current_manifest or statement.
TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(re.escape(t) for t in sorted(COMMON_TERMS, key=lambda x: (-len(x), x)))
    + r")(?![A-Za-z0-9_-])"
)


def transform_plain(text: str, seen: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        term = match.group(0)
        first, later = COMMON_TERMS[term]
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
        converted: list[str] = []
        for i, part in enumerate(parts):
            converted.append(part if i % 2 else transform_plain(part, seen))
        out.append("".join(converted))

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
    print(f"Common-jargon Korean-first pass changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
