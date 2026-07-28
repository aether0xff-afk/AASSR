import unittest

from aassr.gpt2_curriculum import (
    CurriculumBand,
    CurriculumOutcome,
    CurriculumTask,
    LearningProgressScheduler,
)
from aassr.worlds import WorldKind


class GPT2CurriculumTests(unittest.TestCase):
    def test_bootstrap_exposes_all_curriculum_bands(self) -> None:
        scheduler = LearningProgressScheduler(seed=7)
        seen = []
        for _ in range(4):
            task = scheduler.next_task()
            seen.append(task.band)
            scheduler.observe(
                task,
                CurriculumOutcome(success=False, steps=10, step_limit=20),
            )
        self.assertEqual(set(seen), set(CurriculumBand))

    def test_tasks_contain_no_solution_trajectory(self) -> None:
        fields = set(CurriculumTask.__dataclass_fields__)
        self.assertNotIn("solution", fields)
        self.assertNotIn("trajectory", fields)
        self.assertNotIn("actions", fields)

    def test_band_world_mapping_is_capability_based(self) -> None:
        scheduler = LearningProgressScheduler(seed=3)
        tasks = []
        for _ in range(4):
            task = scheduler.next_task()
            tasks.append(task)
            scheduler.observe(
                task,
                CurriculumOutcome(success=True, steps=5, step_limit=20),
            )
        mapping = {task.band: task.world_kind for task in tasks}
        self.assertEqual(mapping[CurriculumBand.FOUNDATION], WorldKind.RANDOM_FLAG)
        self.assertEqual(mapping[CurriculumBand.CONTROL], WorldKind.RANDOM_WALL_FLAG)
        self.assertEqual(mapping[CurriculumBand.COMPOSITION], WorldKind.RANDOM_KEY_DOOR)
        self.assertIn(
            mapping[CurriculumBand.ADVERSARIAL],
            {WorldKind.V2_COMPLEX, WorldKind.LOCKED_BOTTLENECK},
        )


if __name__ == "__main__":
    unittest.main()
