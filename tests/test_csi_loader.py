import numpy as np

from src.preprocess.csi_loader import load_csi_file, list_recordings


def test_load_csi_file_from_npy(tmp_path):
    data = np.random.rand(4, 8) + 1j * np.random.rand(4, 8)
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, data)

    loaded = load_csi_file(sample_path)

    assert loaded.shape == (4, 8)
    assert np.isrealobj(loaded)
    assert np.allclose(loaded, np.abs(data))


def test_load_csi_file_from_txt(tmp_path):
    rows = np.arange(12, dtype=float).reshape(3, 4)
    sample_path = tmp_path / "sample.txt"
    np.savetxt(sample_path, rows)

    loaded = load_csi_file(sample_path)

    assert loaded.shape == (3, 4)
    assert np.isrealobj(loaded)
    assert np.allclose(loaded, rows)


def test_list_recordings(tmp_path):
    (tmp_path / "recordings").mkdir()
    np.save(tmp_path / "recordings" / "a.npy", np.ones((2, 2)))
    np.savetxt(tmp_path / "recordings" / "b.txt", np.ones((2, 2)))

    files = list_recordings(tmp_path / "recordings")

    assert len(files) == 2
    assert all(file.suffix in {".npy", ".txt"} for file in files)

