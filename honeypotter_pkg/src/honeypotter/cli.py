import argparse, pathlib, os, json, math
from tqdm import tqdm
import pandas as pd

from .generator import GenerateDistractor, CategorySpec
from .gamut import ColorSampler

def cmd_generate(args):
    # Input validation
    if args.n_categories <= 0:
        raise ValueError(f"n_categories must be positive, got {args.n_categories}")
    if args.per_class <= 0:
        raise ValueError(f"per_class must be positive, got {args.per_class}")
    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}")
    
    # Validate blur parameters
    if args.pattern_blur_sigma < 0:
        raise ValueError(f"pattern_blur_sigma cannot be negative, got {args.pattern_blur_sigma}")
    if args.global_blur_sigma < 0:
        raise ValueError(f"global_blur_sigma cannot be negative, got {args.global_blur_sigma}")
    
    # Validate gamut file exists
    if args.gamut and not os.path.exists(args.gamut):
        raise FileNotFoundError(f"Gamut file not found: {args.gamut}")
    
    color_sampler = ColorSampler(gamut_path=args.gamut, seed=args.seed)

    families = args.families.split(",") if args.families else ["stripes","checker","dots","blob","solid"]
    if args.enable_primitives and "circle" not in families:
        families.append("circle")
    if args.include_texture_family and "texture" not in families:
        families.append("texture")
    if args.enable_moire and "moire" not in families:
        families.append("moire")

    moire_hues = None
    if args.moire_hues:
        try:
            moire_hues = [float(h.strip()) for h in args.moire_hues.split(",") if h.strip()]
        except ValueError as exc:
            raise ValueError(f"Failed to parse --moire_hues '{args.moire_hues}': {exc}")

    spec = CategorySpec.default(
        families=families,
        n_categories=args.n_categories,
        checker_size_range=(args.checker_size_min, args.checker_size_max),
        checker_color_mode=args.checker_color_mode or args.color_mode,
        stripe_angles=[int(a) for a in args.stripe_angles.split(",")] if args.stripe_angles else [0,45,90],
        stripe_width_range=(args.stripe_width_min, args.stripe_width_max),
        stripe_color_mode=args.stripe_color_mode or args.color_mode,
        dot_radius_range=(args.dot_radius_min, args.dot_radius_max),
        dot_spacing_range=(args.dot_spacing_min, args.dot_spacing_max),
        dot_color_mode=args.dot_color_mode or args.color_mode,
        circle_count_range=(args.circle_count_min, args.circle_count_max),
        circle_radius_range=(args.circle_radius_min, args.circle_radius_max),
        primitive_color_mode=args.primitive_color_mode or args.color_mode,
        blob_scale_range=(args.blob_scale_min, args.blob_scale_max),
        blob_thresh_range=(args.blob_thresh_min, args.blob_thresh_max),
        blob_color_mode=args.blob_color_mode or args.color_mode,
        perlin_scale_range=(args.perlin_scale_min, args.perlin_scale_max),
        pattern_blur_sigma=args.pattern_blur_sigma,
        color_mode=args.color_mode,
        moire_frequency_range=(args.moire_frequency_min, args.moire_frequency_max),
        moire_frequency_delta_range=(args.moire_frequency_delta_min, args.moire_frequency_delta_max),
        moire_angle_range=(args.moire_angle_min, args.moire_angle_max),
        moire_angle_delta_range=(args.moire_angle_delta_min, args.moire_angle_delta_max),
        moire_phase_range=(args.moire_phase_min, args.moire_phase_max),
        moire_grid_size=args.moire_grid_size,
        moire_random_hue=args.moire_random_hue,
        moire_hue_list=moire_hues,
        moire_saturation_range=(args.moire_saturation_min, args.moire_saturation_max),
        moire_value_range=(args.moire_value_min, args.moire_value_max),
    )

    gen = GenerateDistractor(
        image_size=args.image_size,
        spec=spec,
        color_sampler=color_sampler,
        global_blur_sigma=args.global_blur_sigma,
        texture_dir=args.texture_dir,
        include_texture_family=args.include_texture_family,
        texture_mix_prob=args.texture_mix_prob,
        seed=args.seed
    )

    try:
        out_root = pathlib.Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot create output directory '{args.out}': {e}")

    labelmap = {cfg["name"]: idx for idx, cfg in enumerate(gen.categories)}
    with open(out_root / "labelmap.json", "w") as f:
        json.dump(labelmap, f, indent=2)
    with open(out_root / "labels.txt", "w") as f:
        f.write("\n".join([k for k,_ in sorted(labelmap.items(), key=lambda kv: kv[1])]) + "\n")

    for cfg in gen.categories:
        (out_root / cfg["name"]).mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in tqdm(gen.categories, desc="Categories"):
        name = cfg["name"]; cid = labelmap[name]
        for j in range(args.per_class):
            img, _ = gen()
            rel = pathlib.Path(name) / f"{name}_{j:06d}.png"
            img.save(out_root / rel)
            rows.append({
                "filepath": str(rel).replace("\\","/"),
                "label_name": name,
                "label_id": cid,
                "family": cfg["family"],
                "params": json.dumps({k:v for k,v in cfg.items() if k not in ["name","family"]})
            })
    pd.DataFrame(rows).to_csv(out_root / "metadata.csv", index=False)

def build_parser():
    p = argparse.ArgumentParser("Honeypotter CLI")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Generate synthetic distractor categories")
    g.add_argument("--out", required=True)
    g.add_argument("--n_categories", type=int, default=20)
    g.add_argument("--per_class", type=int, default=100)
    g.add_argument("--image_size", type=int, default=224)
    g.add_argument("--families", type=str, default="stripes,checker,dots,blob,solid")
    g.add_argument("--enable_primitives", action="store_true", help="Adds 'circle' family (off by default)")
    g.add_argument("--include_texture_family", action="store_true", help="Adds 'texture' family (requires --texture_dir)")
    g.add_argument("--enable_moire", action="store_true", help="Adds 'moire' interference pattern family")
    g.add_argument("--texture_dir", type=str, default=None)
    g.add_argument("--texture_mix_prob", type=float, default=0.0)

    # global modes
    g.add_argument("--gamut", type=str, help="Path to gamut JSON file (if omitted, uses random RGB)")
    g.add_argument("--color_mode", type=str, default="color", choices=["color","gray","bw","ood"])

    # checker controls
    g.add_argument("--checker_size_min", type=int, default=8)
    g.add_argument("--checker_size_max", type=int, default=32)
    g.add_argument("--checker_color_mode", type=str, default=None)

    # stripes controls
    g.add_argument("--stripe_angles", type=str, default="0,45,90")
    g.add_argument("--stripe_width_min", type=int, default=6)
    g.add_argument("--stripe_width_max", type=int, default=28)
    g.add_argument("--stripe_color_mode", type=str, default=None)

    # dots controls
    g.add_argument("--dot_radius_min", type=int, default=2)
    g.add_argument("--dot_radius_max", type=int, default=8)
    g.add_argument("--dot_spacing_min", type=int, default=8)
    g.add_argument("--dot_spacing_max", type=int, default=28)
    g.add_argument("--dot_color_mode", type=str, default=None)

    # circle primitives
    g.add_argument("--circle_count_min", type=int, default=1)
    g.add_argument("--circle_count_max", type=int, default=5)
    g.add_argument("--circle_radius_min", type=int, default=10)
    g.add_argument("--circle_radius_max", type=int, default=60)
    g.add_argument("--primitive_color_mode", type=str, default=None)

    # blobs
    g.add_argument("--blob_scale_min", type=float, default=12.0)
    g.add_argument("--blob_scale_max", type=float, default=48.0)
    g.add_argument("--blob_thresh_min", type=float, default=0.35)
    g.add_argument("--blob_thresh_max", type=float, default=0.6)
    g.add_argument("--blob_color_mode", type=str, default=None)

    # perlin
    g.add_argument("--perlin_scale_min", type=float, default=8.0)
    g.add_argument("--perlin_scale_max", type=float, default=48.0)

    # blurs
    g.add_argument("--pattern_blur_sigma", type=float, default=0.0)
    g.add_argument("--global_blur_sigma", type=float, default=0.0)

    # moiré options
    g.add_argument("--moire_frequency_min", type=float, default=4.0)
    g.add_argument("--moire_frequency_max", type=float, default=12.0)
    g.add_argument("--moire_frequency_delta_min", type=float, default=0.1)
    g.add_argument("--moire_frequency_delta_max", type=float, default=0.8)
    g.add_argument("--moire_angle_min", type=float, default=0.0)
    g.add_argument("--moire_angle_max", type=float, default=180.0)
    g.add_argument("--moire_angle_delta_min", type=float, default=0.5)
    g.add_argument("--moire_angle_delta_max", type=float, default=6.0)
    g.add_argument("--moire_phase_min", type=float, default=0.0)
    g.add_argument("--moire_phase_max", type=float, default=float(math.tau))
    g.add_argument("--moire_grid_size", type=int, default=16, help="Grid moiré output into N×N cells (set <=1 to disable)")
    g.add_argument("--moire_random_hue", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomize moiré hue (use --no-moire-random-hue to rely on hue list)")
    g.add_argument("--moire_hues", type=str, default=None,
                   help="Comma-separated hue degrees (used when random hue disabled; defaults provided internally)")
    g.add_argument("--moire_saturation_min", type=float, default=0.6)
    g.add_argument("--moire_saturation_max", type=float, default=1.0)
    g.add_argument("--moire_value_min", type=float, default=0.25)
    g.add_argument("--moire_value_max", type=float, default=1.0)

    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    return p

def main():
    try:
        p = build_parser()
        args = p.parse_args()
        if not hasattr(args, "func"):
            p.print_help()
            return
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=__import__('sys').stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=__import__('sys').stderr)
        return 1
    return 0
