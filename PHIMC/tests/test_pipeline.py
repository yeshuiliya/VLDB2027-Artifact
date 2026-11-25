import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from phimc.pipeline import cluster, equal_fusion
from phimc.data import load_multiview
from phimc.metrics import evaluate_clusters


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.views = [self.rng.normal(size=(14, 5)), self.rng.normal(size=(14, 8))]
        self.labels = np.arange(14) % 2

    def test_standard_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "standard.mat"
            savemat(filename, {"X1": self.views[0], "X2": self.views[1].T,
                               "Y": self.labels.reshape(-1, 1)})
            data = load_multiview(filename)
            self.assertEqual(data.dimensions, [5, 8])
            self.assertEqual(data.n_views, 2)
            self.assertTrue(np.isfinite(data.views[1]).all())

    def test_classic_cell_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "cell.mat"
            cells = np.empty((1, 2), dtype=object)
            cells[0, 0], cells[0, 1] = self.views
            savemat(filename, {"X": cells, "Y": self.labels.reshape(-1, 1)})
            data = load_multiview(filename, normalize_input=False)
            self.assertEqual(data.dimensions, [5, 8])

    def test_hdf5_matlab_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "v73.mat"
            with h5py.File(filename, "w") as f:
                g = f.create_group("#refs#")
                a = g.create_dataset("a", data=self.views[0].T)
                b = g.create_dataset("b", data=self.views[1].T)
                x = f.create_dataset("X", shape=(1, 2), dtype=h5py.ref_dtype)
                x[0, 0], x[0, 1] = a.ref, b.ref
                f.create_dataset("Y", data=self.labels.reshape(1, -1))
            data = load_multiview(filename)
            self.assertEqual(data.dimensions, [5, 8])

    def test_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "example.npz"
            np.savez_compressed(filename, X1=self.views[0], X2=self.views[1], Y=self.labels)
            data = load_multiview(filename)
            self.assertEqual(data.n_samples, 14)

    def test_feature_fusion(self):
        fused = equal_fusion(self.views)
        self.assertEqual(fused.shape, (14, 13))
        self.assertGreaterEqual(fused.min(), 0)
        self.assertLessEqual(fused.max(), 1)

    def test_metrics_permuted_ids(self):
        actual = np.array([2, 2, 5, 5, 2, 5])
        predicted = np.array([7, 7, 9, 9, 7, 9])
        scores = evaluate_clusters(actual, predicted)
        self.assertAlmostEqual(scores["ACC"], 1)
        self.assertAlmostEqual(scores["NMI"], 1)
        self.assertAlmostEqual(scores["ARI"], 1)

    def test_small_clustering(self):
        z1 = np.vstack([np.tile([0., 1., 2.], (8, 1)),
                        np.tile([2., 0., 1.], (8, 1))]).astype(np.float32)
        z2 = z1 * 2
        labels = np.array([0] * 8 + [1] * 8)
        _, preds, scores = cluster([z1, z2], labels, n_clusters=2)
        self.assertEqual(preds.shape, (16,))
        self.assertAlmostEqual(scores["ACC"], 1)


if __name__ == "__main__":
    unittest.main()
