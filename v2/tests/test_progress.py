import io
import unittest

from aassr.progress import ProgressTracker


class ProgressTrackerTests(unittest.TestCase):
    def test_progress_tracker_prints_percent_and_eta(self) -> None:
        stream = io.StringIO()
        tracker = ProgressTracker(label="demo", total=2, stream=stream)

        tracker.advance()
        tracker.advance()

        output = stream.getvalue()
        self.assertIn("[demo]", output)
        self.assertIn("100.0%", output)
        self.assertIn("eta=", output)

    def test_disabled_tracker_is_quiet(self) -> None:
        stream = io.StringIO()
        tracker = ProgressTracker(label="quiet", total=1, enabled=False, stream=stream)

        tracker.advance()

        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
