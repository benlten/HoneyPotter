from pathlib import Path

from honeypotter.cli import build_parser


def test_patterns_only_generates_flat_directory(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "generate",
        "--out",
        str(tmp_path),
        "--families",
        "moire",
        "--enable_moire",
        "--n_categories",
        "2",
        "--per_class",
        "3",
        "--image_size",
        "32",
        "--patterns_only",
        "--seed",
        "123",
    ])

    args.func(args)

    png_files = sorted(Path(tmp_path).glob("*.png"))
    assert len(png_files) == 6
    assert not (Path(tmp_path) / "labelmap.json").exists()
    assert not (Path(tmp_path) / "metadata.csv").exists()
