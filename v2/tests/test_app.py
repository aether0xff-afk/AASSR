import unittest

from aassr.dashboard import (
    binding_rows,
    candidate_rows,
    imagination_score_rows,
    implementation_status_rows,
    knowledge_rows,
    paper_project_comparison_rows,
    policy_probability_rows,
)
from aassr import DMPConfig, GridWorld, GridWorldDMP, ImaginationCycle, KK
from aassr.imagination import ImaginationConfig
from aassr.policy import PolicyABC
from aassr.prophecy import ProphecyPrediction, TableProphecyModel


class FixedProphecy(TableProphecyModel):
    def predict(self, state_signature, candidate):
        return ProphecyPrediction(
            kk_probs={kk: 0.0 for kk in KK},
            error_prob=0.1,
            flag_prob=0.2,
        )


class StreamlitAppTests(unittest.TestCase):
    def test_knowledge_rows_formats_mixed_kv_values_as_strings(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        rows = knowledge_rows(dmp)

        self.assertTrue(rows)
        self.assertTrue(all(isinstance(row["KV"], str) for row in rows))

    def test_candidate_rows_expose_what_how_where_and_binding(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        rows = candidate_rows(dmp)

        self.assertTrue(rows)
        self.assertIn("WHAT", rows[0])
        self.assertIn("HOW", rows[0])
        self.assertIn("WHERE", rows[0])
        self.assertIn("bound_KV", rows[0])
        self.assertIn("executable_action", rows[0])

    def test_binding_rows_show_kk_slot_to_kv_to_command(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        rows = binding_rows(dmp)

        self.assertTrue(rows)
        self.assertIn("KK slot", rows[0])
        self.assertIn("KV bound", rows[0])
        self.assertIn("generated command", rows[0])
        self.assertTrue(any(row["KK slot"] == "KK_UNKNOWN_NEIGHBOR" for row in rows))

    def test_policy_probability_rows_show_policyabc_tables(self) -> None:
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            scorer=PolicyABC.uniform_gridworld(seed=0),
        )

        rows = policy_probability_rows(dmp)

        self.assertTrue(rows)
        self.assertTrue(any(row["axis"] == "WHAT" for row in rows))
        self.assertTrue(any(row["axis"] == "HOW" for row in rows))
        self.assertTrue(any(row["axis"] == "WHERE" for row in rows))

    def test_imagination_score_rows_show_c3_candidate_scores(self) -> None:
        prophecy = FixedProphecy()
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            scorer=PolicyABC.uniform_gridworld(seed=0),
            prophecy=prophecy,
            imagination=ImaginationCycle(prophecy, ImaginationConfig(policy_prior_weight=0.0)),
            config=DMPConfig(use_prophecy=True, use_imagination=True),
        )
        selected = dmp.choose_candidate()
        result = dmp.execute(selected)

        rows = imagination_score_rows(result)

        self.assertTrue(rows)
        self.assertIn("score", rows[0])
        self.assertIn("flag_prob", rows[0])
        self.assertIn("rollout_value", rows[0])
        self.assertIn("rollout_depth", rows[0])
        self.assertTrue(any(row["selected"] for row in rows))

    def test_paper_project_comparison_rows_show_parallel_mapping(self) -> None:
        rows = paper_project_comparison_rows()

        self.assertTrue(rows)
        self.assertIn("Original APASSR / prior setting", rows[0])
        self.assertIn("This project", rows[0])
        self.assertTrue(any(row["axis"] == "Prophecy" for row in rows))
        self.assertTrue(any("GridWorld" in row["This project"] for row in rows))

    def test_implementation_status_rows_show_full_pipeline(self) -> None:
        rows = implementation_status_rows()

        self.assertTrue(rows)
        self.assertTrue(any(row["module"] == "ExperimentRunner" for row in rows))
        self.assertTrue(any(row["module"] == "Analysis" for row in rows))
        self.assertTrue(any(row["status"] == "Pending run" for row in rows))


if __name__ == "__main__":
    unittest.main()
