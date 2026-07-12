import tempfile
import unittest
from pathlib import Path

from aassr import CellKind
from aassr.v2_compare import run_v2_comparison
from aassr.worlds import WorldKind, make_world


class V2ComplexTests(unittest.TestCase):
    def test_v2_complex_world_has_larger_map_and_multiple_objects(self) -> None:
        world = make_world(WorldKind.V2_COMPLEX, seed=0)
        kinds = list(world.cells.values())

        self.assertGreaterEqual(world.width, 9)
        self.assertGreaterEqual(world.height, 6)
        self.assertGreaterEqual(kinds.count(CellKind.WALL), 8)
        self.assertGreaterEqual(kinds.count(CellKind.KEY), 2)
        self.assertGreaterEqual(kinds.count(CellKind.DOOR), 2)
        self.assertGreaterEqual(kinds.count(CellKind.HINT), 2)
        self.assertEqual(kinds.count(CellKind.FLAG), 1)

    def test_locked_bottleneck_world_has_mandatory_doors(self) -> None:
        world = make_world(WorldKind.LOCKED_BOTTLENECK, seed=0)
        kinds = list(world.cells.values())

        self.assertGreaterEqual(world.width, 10)
        self.assertGreaterEqual(kinds.count(CellKind.DOOR), 2)
        self.assertGreaterEqual(kinds.count(CellKind.KEY), 1)
        self.assertGreaterEqual(kinds.count(CellKind.HINT), 2)
        self.assertEqual(kinds.count(CellKind.FLAG), 1)

    def test_v2_comparison_writes_all_baselines_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_v2_comparison(
                episodes=1,
                seeds=1,
                step_limit=20,
                output_dir=tmpdir,
            )

            for condition in ("C0", "C1", "C2", "C3", "C4", "C5", "QLEARN", "DQN_PARTIAL", "ORACLE_MDP"):
                self.assertTrue((output / condition / "gridworld_episodes.csv").exists(), condition)
                self.assertTrue((output / condition / "gridworld_summary.csv").exists(), condition)
            self.assertTrue((output / "analysis" / "summary_table.csv").exists())
            self.assertTrue((output / "analysis" / "figure_success_rate.png").exists())


if __name__ == "__main__":
    unittest.main()
