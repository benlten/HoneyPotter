import numpy as np

from honeypotter.generator import CategorySpec, GenerateDistractor


def test_moire_generation_grid_uniform_cells():
    spec = CategorySpec.default(
        families=["moire"],
        n_categories=1,
        moire_grid_size=16,
        moire_random_hue=True,
    )
    gen = GenerateDistractor(image_size=128, spec=spec, seed=7)
    img, name = gen()

    assert name.startswith("moire_")
    arr = np.array(img)
    assert arr.shape == (128, 128, 3)

    cfg = gen.categories[0]
    assert cfg.get("grid_size") == 16

    y_edges = np.linspace(0, arr.shape[0], cfg["grid_size"] + 1, dtype=int)
    x_edges = np.linspace(0, arr.shape[1], cfg["grid_size"] + 1, dtype=int)
    for yi in range(cfg["grid_size"]):
        for xi in range(cfg["grid_size"]):
            block = arr[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]]
            reference = block[0, 0]
            assert np.all(block == reference)


def test_moire_hue_list_selection():
    hue_list = [15.0, 195.0, 305.0]
    spec = CategorySpec.default(
        families=["moire"],
        n_categories=6,
        moire_grid_size=0,
        moire_random_hue=False,
        moire_hue_list=hue_list,
    )
    gen = GenerateDistractor(image_size=64, spec=spec, seed=11)

    hues = {round(cfg["hue"], 1) for cfg in gen.categories}
    assert hues.issubset({round(h, 1) for h in hue_list})
    assert len(hues) >= 2  # ensure we covered more than one hue from the list

