from honeypotter import GenerateDistractor, CategorySpec, Palette

def main():
    spec_pal = CategorySpec.default(
        families=["checker","stripes","dots","blob","solid","circle","perlin"],
        n_categories=12,
        checker_size_range=(6, 28),
        stripe_angles=[0,45,90],
        stripe_width_range=(6,24),
        color_mode="palette",
        pattern_blur_sigma=2.0
    )
    spec_ood = CategorySpec.default(
        families=spec_pal.families,
        n_categories=12,
        checker_size_range=(6, 28),
        stripe_angles=[0,45,90],
        stripe_width_range=(6,24),
        color_mode="ood",
        pattern_blur_sigma=2.0
    )
    gen_pal = GenerateDistractor(image_size=224, spec=spec_pal, palette=Palette.imagenet_default(), seed=7)
    gen_ood = GenerateDistractor(image_size=224, spec=spec_ood, palette=Palette.imagenet_default(), seed=7)
    for i in range(3):
        img, name = gen_pal(); img.save(f"pal_{i}_{name}.png")
        img, name = gen_ood(); img.save(f"ood_{i}_{name}.png")
    print("Wrote 6 samples.")
if __name__ == "__main__":
    main()
