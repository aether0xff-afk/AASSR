from __future__ import annotations

import re
from pathlib import Path

WIKI = Path("wiki")

EXTRA_TERMS: dict[str, tuple[str, str]] = {
    "real": ("[실제 환경에서 관측된(real)](Research-Jargon-Guide)", "실제"),
    "imagined": ("[모델이 상상한(imagined)](Research-Jargon-Guide)", "가상"),
    "external": ("[환경이 주는 외부(external)](Terminology-Guide)", "환경이 주는 외부"),
    "outcome": ("[환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)", "환경 결과"),
    "probability": ("[확률(probability)](Stochasticity-Uncertainty-and-Probability)", "확률"),
    "expectation": ("[확률 기댓값(expectation)](Chance-and-Decision-Nodes)", "확률 기댓값"),
    "backup": ("[미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)", "가치 되돌림 계산"),
    "concrete": ("[실제 개체를 구분하는(concrete)](State-Representation)", "실제 개체를 구분하는"),
    "structural": ("[구조 기반(structural)](Relational-Representation-and-Generalization)", "구조 기반"),
    "template": ("[재사용 가능한 틀(template)](Skills)", "재사용 가능한 틀"),
    "role": ("[역할(role)](Relational-Representation-and-Generalization)", "역할"),
    "estimate": ("[추정값(estimate)](Value-Functions-and-Bellman-Equation)", "추정값"),
    "estimator": ("[값을 추정하는 모델(estimator)](Terminology-Guide)", "값을 추정하는 모델"),
    "comparison": ("[비교(comparison)](Ablation-Benchmarking-and-Reproducibility)", "비교"),
    "causal": ("[인과적으로 공정한(causal)](Causality-Leakage-and-Evaluation)", "인과적으로 공정한"),
    "invariant": ("[수정해도 유지되어야 하는 성질(invariant)](Research-Jargon-Guide)", "불변 조건"),
    "semantic": ("[의미 기준(semantic)](State-Representation)", "의미 기준"),
    "descriptor": ("[상태를 요약한 표현(descriptor)](State-Representation)", "상태 요약 표현"),
    "feature": ("[학습에 사용하는 특징(feature)](Terminology-Guide)", "학습 특징"),
    "features": ("[학습에 사용하는 특징(features)](Terminology-Guide)", "학습 특징"),
    "score": ("[평가 점수(score)](Terminology-Guide)", "평가 점수"),
    "scorer": ("[점수를 계산하는 평가기(scorer)](Terminology-Guide)", "점수 평가기"),
    "threshold": ("[판정 기준값(threshold)](Terminology-Guide)", "판정 기준값"),
    "coverage": ("[데이터가 어느 영역까지 포함하는지(coverage)](Critic-Support-and-OOD)", "데이터 포함 범위"),
    "confidence": ("[예측 신뢰 정도(confidence)](Calibration)", "예측 신뢰 정도"),
    "distribution": ("[확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)", "분포"),
    "mode": ("[서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)", "결과 유형"),
    "modes": ("[서로 다른 결과 유형(modes)](Mixture-Ensemble-and-Calibration)", "결과 유형"),
    "branch": ("[갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)", "결과 경로"),
    "branches": ("[갈라진 결과 경로(branches)](Chance-and-Decision-Nodes)", "결과 경로"),
    "candidate": ("[선택 후보(candidate)](Terminology-Guide)", "선택 후보"),
    "candidates": ("[선택 후보(candidates)](Terminology-Guide)", "선택 후보"),
    "ranking": ("[후보 순위(ranking)](Policy)", "후보 순위"),
    "baseline comparison": ("[비교 기준 모델과의 비교(baseline comparison)](Ablation-Benchmarking-and-Reproducibility)", "비교 기준 모델과의 비교"),
    "blind": ("[결과를 미리 보지 않는 비공개 평가(blind)](Ablation-Benchmarking-and-Reproducibility)", "비공개"),
}

PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(re.escape(t) for t in sorted(EXTRA_TERMS, key=lambda x: (-len(x), x)))
    + r")(?![A-Za-z0-9_-])"
)


def transform_plain(text: str, seen: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        term = match.group(0)
        first, later = EXTRA_TERMS[term]
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
    print(f"Remaining-jargon pass changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
