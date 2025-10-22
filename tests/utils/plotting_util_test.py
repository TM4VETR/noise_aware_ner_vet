import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from utils.plotting_util import plot_and_save


def test_plot_and_save_creates_file(tmp_path):
    out_dir = tmp_path / "figs"
    out_file = out_dir / "example"

    # create a simple plot
    plt.figure()
    plt.plot([0, 1, 2], [0, 1, 0])
    plt.plot([1, 2, 3], [0, 1, 0])

    # save with a kwarg (dpi)
    plot_and_save(str(out_file), dpi=300)

    # assertions
    for ext in ["png", "pdf"]:
        file_path = out_file.with_suffix(f".{ext}")
        assert file_path.exists(), f"Output file {file_path} was not created"
        assert file_path.stat().st_size > 0, f"Output file {file_path} is empty"
