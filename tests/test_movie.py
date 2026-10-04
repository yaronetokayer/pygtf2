"""Movie API tests: render real frames, intercept only the ffmpeg invocation."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import pygtf2 as gtf
from pygtf2.plot import snapshot


class MovieTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("PYGTF_TEST_TMPDIR", "."))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / "Model00000"
        self.model.mkdir()
        (self.model / "snapshot_conversion.txt").write_text(
            "index time time_Gyr step\n0 0 0 0\n1 1 1 1\n")
        header = "log_r log_rmid m_tot rho_tot v2_tot eta lgr[stars] lgrm[stars] m[stars] rho[stars] v2[stars]\n"
        for i in range(2):
            (self.model / f"profile_{i}.dat").write_text(header +
                f"0 -0.3 1 {2+i} 1 0.1 0 -0.3 1 {2+i} 1\n" +
                "1 0.7 2 0.1 0.5 0.2 1 0.7 2 0.1 0.5\n")

    def encode(self, command, **kwargs):
        self.assertEqual(command[0], "ffmpeg")
        frames = sorted((self.model / "temp_images").glob("*.png"))
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(p.read_bytes().startswith(b"\x89PNG") for p in frames))
        self.assertTrue(all(p.stat().st_size > 1000 for p in frames))
        self.assertEqual(Path(command[-1]), self.model / "movie.mp4")

    def test_no_insets_serial_and_parallel_without_time_series(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel), patch.object(
                snapshot.subprocess, "run", side_effect=self.encode
            ) as encode, patch.object(snapshot.os, "cpu_count", return_value=None):
                gtf.make_movie(0, base_dir=str(self.base), insets=False,
                               profiles=["rho", "v2"], xaxis="m", parallel=parallel)
                encode.assert_called_once()
                self.assertFalse((self.model / "temp_images").exists())

    def test_insets_and_radii_serial_and_parallel(self):
        (self.model / "time_evolution.txt").write_text(
            "time rho_c_tot r_c\n0 2 0.5\n1 3 0.4\n")
        for parallel in (False, True):
            with self.subTest(parallel=parallel), patch.object(
                snapshot.subprocess, "run", side_effect=self.encode
            ), patch.object(snapshot.os, "cpu_count", return_value=4):
                gtf.make_movie(0, base_dir=str(self.base), profiles=["rho", "v2"],
                               add_radii=["r_c"], parallel=parallel)

    def test_fixed_limits_and_state_entry_point(self):
        config = gtf.Config(io={"base_dir": str(self.base), "model_no": 0})
        state = object.__new__(gtf.State)
        state.config = config
        with patch.object(snapshot, "plot_profile", wraps=snapshot.plot_profile) as plot, \
             patch.object(snapshot.subprocess, "run", side_effect=self.encode):
            state.make_movie(profiles="rho", insets=[None], parallel=False)
        self.assertEqual(plot.call_count, 2)
        self.assertEqual(plot.call_args_list[0].kwargs["axislims"],
                         plot.call_args_list[1].kwargs["axislims"])

    def test_exports_and_invalid_options(self):
        self.assertFalse(hasattr(gtf, "make_movie_deluxe"))
        self.assertIn("make_movie", gtf.__all__)
        for options in ({"profiles": []}, {"insets": True},
                        {"profiles": ["kn"]}, {"insets": [None]},
                        {"xaxis": "invalid"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                gtf.make_movie(0, base_dir=str(self.base), parallel=False, **options)


if __name__ == "__main__":
    unittest.main()
