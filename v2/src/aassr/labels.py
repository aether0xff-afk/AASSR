from __future__ import annotations


CONDITION_LABELS = {
    "C0": "C0 Random",
    "C1": "C1 PolicyABC",
    "C2": "C2 PolicyABC + Prophecy",
    "C3": "C3 PolicyABC + Prophecy + Imagination",
    "C4": "C4 PolicyABC + Sequence Prophecy variant + Imagination",
    "QLEARN": "Q-learning baseline",
    "DQN_PARTIAL": "DQN partial-observation baseline",
    "ORACLE_MDP": "Oracle MDP, full-map upper bound",
    "A1_TABLE_C3": "A1 Table Prophecy C3",
    "A1_TRANSFORMER_C3": "A1 Transformer Prophecy C3",
    "A2_REWARD_ON": "A2 Prophecy reward on",
    "A2_REWARD_OFF": "A2 Prophecy reward off",
}


def condition_label(condition: str) -> str:
    if condition.startswith("A3_D") and "_B" in condition:
        return condition.replace("A3_D", "A3 depth ").replace("_B", ", branch ")
    return CONDITION_LABELS.get(condition, condition)
