from honeypotter import GenerateDistractor, CategorySpec
from honeypotter.gamut import ColorSampler


def test_ood_smoke():
    spec = CategorySpec.default(
        n_categories=5,
        families=["checker", "stripes", "dots", "blob", "solid", "circle", "perlin"],
        color_mode="ood",
        pattern_blur_sigma=1.0,
    )
    sampler = ColorSampler(seed=0)
    gen = GenerateDistractor(image_size=64, spec=spec, color_sampler=sampler, seed=0)
    img, name = gen()
    assert img.size == (64, 64)
    assert isinstance(name, str)
