import unittest

from aassr import CellKind, GridWorld, GridWorldDMP
from aassr.visualization import render_candidates, render_dmp_mermaid, render_dmp_state, render_grid


class VisualizationTests(unittest.TestCase):
    def test_grid_render_marks_agent_and_frontier(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        rendered = render_grid(dmp)

        self.assertIn("A", rendered)
        self.assertIn("?", rendered)

    def test_dmp_state_includes_knowledge_and_candidates(self) -> None:
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1), cells={(1, 0): CellKind.WALL})
        )
        dmp.execute(next(iter(dmp.generate_candidates())))

        rendered = render_dmp_state(dmp)

        self.assertIn("GridWorld", rendered)
        self.assertIn("Knowledge Storage", rendered)
        self.assertIn("Action Candidates", rendered)

    def test_candidate_render_shows_binding(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        rendered = render_candidates(dmp.generate_candidates())

        self.assertIn("INSPECT_CELL", rendered)
        self.assertIn("KK_CURRENT_POS", rendered)

    def test_mermaid_render_documents_core_loop(self) -> None:
        rendered = render_dmp_mermaid()

        self.assertIn("Knowledge Storage", rendered)
        self.assertIn("KK Slot Binding", rendered)
        self.assertIn("Policy / Prophecy / Imagination", rendered)


if __name__ == "__main__":
    unittest.main()
