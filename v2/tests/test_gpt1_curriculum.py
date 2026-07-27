import unittest

from aassr.curriculum import CurriculumBand, LearningProgressScheduler
from aassr.worlds import WorldKind


class CurriculumTests(unittest.TestCase):
    def test_scheduler_exposes_all_bands_before_specializing(self) -> None:
        scheduler = LearningProgressScheduler(seed=7)

        tasks = [scheduler.next_task() for _ in range(4)]

        self.assertEqual({task.band for task in tasks}, set(CurriculumBand))

    def test_curriculum_never_contains_solution_trajectory(self) -> None:
        scheduler = LearningProgressScheduler(seed=3)

        task = scheduler.next_task()

        self.assertFalse(hasattr(task, "actions"))
        self.assertFalse(hasattr(task, "solution"))

    def test_band_maps_to_increasingly_compositional_worlds(self) -> None:
        scheduler = LearningProgressScheduler(seed=11)
        observed = {}
        for _ in range(4):
            task = scheduler.next_task()
            observed[task.band] = task.world_kind

        self.assertEqual(observed[CurriculumBand.FOUNDATION], WorldKind.RANDOM_FLAG)
        self.assertEqual(observed[CurriculumBand.CONTROL], WorldKind.RANDOM_WALL_FLAG)
        self.assertEqual(observed[CurriculumBand.COMPOSITION], WorldKind.RANDOM_KEY_DOOR)
        self.assertIn(
            observed[CurriculumBand.ADVERSARIAL],
            {WorldKind.V2_COMPLEX, WorldKind.LOCKED_BOTTLENECK},
        )


if __name__ == "__main__":
    unittest.main()
