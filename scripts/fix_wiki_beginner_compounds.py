from __future__ import annotations

from pathlib import Path

WIKI = Path("wiki")

REPLACEMENTS = {
    "same [체크포인트(checkpoint)](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "same [체크포인트](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "same 체크포인트": "[같은 체크포인트(same checkpoint)](Experiments)",
    "Same [체크포인트(checkpoint)](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "Same 체크포인트": "[같은 체크포인트(same checkpoint)](Experiments)",
    "Critic 데이터 근거": "가치 평가 데이터 근거",
    "희소 보상([희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment))": "[희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)",
    "학습 분포 밖([학습 분포 밖(OOD)](Critic-Support-and-OOD), Out-of-Distribution)": "[학습 분포 밖(OOD, Out-of-Distribution)](Critic-Support-and-OOD)",
    "보상([보상(reward)](Sparse-Reward-and-Credit-Assignment))": "[보상(reward)](Sparse-Reward-and-Credit-Assignment)",
    "누적 보상([누적 보상(return)](Value-Functions-and-Bellman-Equation))": "[누적 보상(return)](Value-Functions-and-Bellman-Equation)",
    "Q값([Q값(Q-value)](Value-Functions-and-Bellman-Equation))": "[Q값(Q-value)](Value-Functions-and-Bellman-Equation)",
    "관측([관측(observation)](MDP-and-POMDP))": "[관측(observation)](MDP-and-POMDP)",
    "표현([표현(representation)](Relational-Representation-and-Generalization))": "[표현(representation)](Relational-Representation-and-Generalization)",
    "외부 제한 종료([외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries))": "[외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)",
    "현재 세대([현재 세대(current-generation)](Current-Status))": "[현재 세대(current-generation)](Current-Status)",
    "현재 실행 구조([현재 실행 구조(current runtime)](Current-Status))": "[현재 실행 구조(current runtime)](Current-Status)",
    "현재 구조([현재 구조(current architecture)](Current-Status))": "[현재 구조(current architecture)](Current-Status)",
    "진단 실험([진단 실험(diagnostic)](Evidence-Matrix))": "[진단 실험(diagnostic)](Evidence-Matrix)",
    "증거([증거(evidence)](Evidence-Matrix))": "[증거(evidence)](Evidence-Matrix)",
    "연구 주장([연구 주장(claim)](Evidence-Matrix))": "[연구 주장(claim)](Evidence-Matrix)",
}


def main() -> None:
    changed = []
    for path in sorted(WIKI.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in REPLACEMENTS.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.as_posix())
    print(f"Compound terminology fixes changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
