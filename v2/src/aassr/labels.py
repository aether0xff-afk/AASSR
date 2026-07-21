from __future__ import annotations


CONDITION_LABELS = {
    "C0": "C0 Random",
    "C1": "C1 PolicyABC",
    "C2": "C2 PolicyABC + Prophecy",
    "C3": "C3 PolicyABC + Prophecy + Imagination",
    "C4": "C4 PolicyABC + Sequence Prophecy variant + Imagination",
    "C5": "C5 Improved APASSR",
    "APASSR_FULL": "APASSR_FULL Predicted-state Imagination",
    "APASSR_FULL_CAL": "APASSR_FULL_CAL Calibrated Imagination",
    "QLEARN": "Q-learning baseline",
    "DQN_PARTIAL": "DQN partial-observation baseline",
    "ORACLE_MDP": "Oracle MDP, full-map upper bound",
    "A1_TABLE_C3": "A1 Table Prophecy C3",
    "A1_TRANSFORMER_C3": "A1 Transformer Prophecy C3",
    "A2_REWARD_ON": "A2 Prophecy reward on",
    "A2_REWARD_OFF": "A2 Prophecy reward off",
    "A4_FULL_C3": "A4 Full C3",
    "A4_NO_DEPENDENCY": "A4 no dependency bonus",
    "A4_NO_REPEAT_PENALTY": "A4 no repeat penalty",
    "A4_NO_POLICY_PRIOR": "A4 no policy prior",
    "A4_NO_ROLLOUT_VALUE": "A4 no rollout value",
    "A4_ONE_STEP_NO_DEP": "A4 one-step no dependency",
    "A5_FULL_C3": "A5 Full C3",
    "A5_NO_KNOWLEDGE_GAIN": "A5 no knowledge-gain score",
    "A5_NO_FLAG_PROB": "A5 no flag-probability score",
    "A5_NO_ERROR_AVOIDANCE": "A5 no error-avoidance score",
}


def condition_label(condition: str) -> str:
    if condition.startswith("A3_D") and "_B" in condition:
        return condition.replace("A3_D", "A3 depth ").replace("_B", ", branch ")
    return CONDITION_LABELS.get(condition, condition)
