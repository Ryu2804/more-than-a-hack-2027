import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location("method_replay_export", Path(__file__).with_name("prepare_method_replay.py"))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()

    def test_relative_output_is_rejected(self):
        with self.assertRaises(ValueError):
            export.outside_output(Path("replay.json"), self.repo, False)

    def test_repository_output_is_rejected_even_with_force(self):
        with self.assertRaises(ValueError):
            export.outside_output(self.repo / "replay.json", self.repo, True)

    def test_existing_output_requires_force(self):
        output = self.root / "replay.json"
        output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export.outside_output(output, self.repo, False)
        self.assertEqual(export.outside_output(output, self.repo, True), output)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_symlink_parent_cannot_redirect_into_repository(self):
        link = self.root / "redirect"
        try:
            link.symlink_to(self.repo, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            export.outside_output(link / "replay.json", self.repo, True)

    def test_alternate_repo_cannot_allow_output_in_exporter_repository(self):
        actual_repo = Path(export.__file__).resolve().parents[2]
        with self.assertRaises(ValueError):
            export.outside_output(actual_repo / "replay.json", self.repo, True)


class SampleMappingTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("TOTAL_FRAMES", 40), ("FIRST_FRAME", 25), ("WINDOW_FRAMES", 10)):
            patcher = patch.object(export, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.data = {"sampled": np.array([5, 10, 15, 25, 30, 35]),
                     "blocks": np.array([[0, 3], [3, 6]]),
                     "blocks25": np.array([[5, 16], [25, 36]])}

    def test_window_uses_source_frame_mapping_not_sample_zero(self):
        np.testing.assert_array_equal(export.sampled_window(self.data), [3, 4])

    def test_sampled_block_disagreement_is_rejected(self):
        self.data["sampled"][3] = 24
        with self.assertRaises(ValueError):
            export.sampled_window(self.data)

    def test_window_crossing_tracking_gap_is_rejected(self):
        with patch.object(export, "FIRST_FRAME", 15):
            with self.assertRaises(ValueError):
                export.sampled_window(self.data)

    def test_nonfinite_coordinates_become_null_without_clipping_finite_values(self):
        self.assertIsNone(export.coordinate(np.array([np.nan, 0.5])))
        self.assertIsNone(export.coordinate(np.array([0.5, np.inf])))
        self.assertEqual(export.coordinate(np.array([-0.02, 1.04])), [-0.02, 1.04])


if __name__ == "__main__":
    unittest.main()
