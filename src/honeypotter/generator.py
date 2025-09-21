import os, glob, json, math, random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .gamut import ColorSampler


DEFAULT_MOIRE_HUES = [15.0, 45.0, 90.0, 180.0, 220.0, 300.0]

@dataclass
class CategorySpec:
    """Configuration for generating distractor categories.
    
    Defines the types of patterns, their parameters, and generation settings
    for creating synthetic distractor images.
    
    Attributes:
        families: List of pattern types to generate (e.g., "checker", "stripes", "dots")
        n_categories: Total number of distractor categories to create
    """
    families: List[str] = field(default_factory=lambda: ["stripes","checker","dots","blob","solid"])
    n_categories: int = 20
    checker_size_range: Tuple[int,int] = (8, 32)
    checker_color_mode: Optional[str] = None
    stripe_angles: List[int] = field(default_factory=lambda: [0, 30, 45, 60, 90])
    stripe_width_range: Tuple[int,int] = (6, 28)
    stripe_color_mode: Optional[str] = None
    dot_radius_range: Tuple[int,int] = (2, 8)
    dot_spacing_range: Tuple[int,int] = (8, 28)
    dot_color_mode: Optional[str] = None
    circle_count_range: Tuple[int,int] = (1, 5)
    circle_radius_range: Tuple[int,int] = (10, 60)
    primitive_color_mode: Optional[str] = None
    blob_scale_range: Tuple[float,float] = (12.0, 48.0)
    blob_thresh_range: Tuple[float,float] = (0.35, 0.6)
    blob_color_mode: Optional[str] = None
    perlin_scale_range: Tuple[float,float] = (8.0, 48.0)
    pattern_blur_sigma: float = 0.0
    color_mode: str = "color"   # color|gray|bw|ood
    moire_frequency_range: Tuple[float, float] = (4.0, 12.0)
    moire_frequency_delta_range: Tuple[float, float] = (0.1, 0.8)
    moire_angle_range: Tuple[float, float] = (0.0, 180.0)
    moire_angle_delta_range: Tuple[float, float] = (0.5, 6.0)
    moire_phase_range: Tuple[float, float] = (0.0, math.tau)
    moire_grid_size: Optional[int] = 16
    moire_random_hue: bool = True
    moire_hue_list: Optional[List[float]] = None
    moire_saturation_range: Tuple[float, float] = (0.6, 1.0)
    moire_value_range: Tuple[float, float] = (0.25, 1.0)

    @staticmethod
    def default(**kwargs):
        return CategorySpec(**kwargs)

    def instantiate(self, rng: Optional[random.Random]=None) -> List[Dict]:
        rng = rng or random.Random()
        cats = []
        for _ in range(self.n_categories):
            fam = rng.choice(self.families)
            cfg = {"family": fam}
            if fam == "solid":
                cfg.update({"color_mode": self.color_mode})
            elif fam == "checker":
                cfg.update({"tile": rng.randint(*self.checker_size_range),
                            "color_mode": self.checker_color_mode or self.color_mode})
            elif fam == "stripes":
                cfg.update({"angle": rng.choice(self.stripe_angles),
                            "width": rng.randint(*self.stripe_width_range),
                            "color_mode": self.stripe_color_mode or self.color_mode})
            elif fam == "dots":
                cfg.update({"radius": rng.randint(*self.dot_radius_range),
                            "spacing": rng.randint(*self.dot_spacing_range),
                            "color_mode": self.dot_color_mode or self.color_mode})
            elif fam == "circle":
                cfg.update({"count": rng.randint(*self.circle_count_range),
                            "rmin": self.circle_radius_range[0],
                            "rmax": self.circle_radius_range[1],
                            "color_mode": self.primitive_color_mode or self.color_mode})
            elif fam == "blob":
                cfg.update({"scale": rng.uniform(*self.blob_scale_range),
                            "thresh": rng.uniform(*self.blob_thresh_range),
                            "color_mode": self.blob_color_mode or self.color_mode})
            elif fam == "perlin":
                cfg.update({"scale": rng.uniform(*self.perlin_scale_range),
                            "color_mode": self.color_mode})
            elif fam == "texture":
                cfg.update({"mode": "texture", "color_mode": self.color_mode})
            elif fam == "moire":
                grid = None
                if self.moire_grid_size and self.moire_grid_size > 1:
                    grid = int(self.moire_grid_size)
                hue_list = self.moire_hue_list or DEFAULT_MOIRE_HUES
                if self.moire_random_hue:
                    hue = rng.uniform(0.0, 360.0) if not self.moire_hue_list else rng.choice(hue_list)
                    hue_source = "random" if not self.moire_hue_list else "list"
                else:
                    hue = rng.choice(hue_list)
                    hue_source = "list" if self.moire_hue_list else "default"
                cfg.update({
                    "frequency": rng.uniform(*self.moire_frequency_range),
                    "frequency_delta": rng.uniform(*self.moire_frequency_delta_range),
                    "angle": rng.uniform(*self.moire_angle_range),
                    "angle_delta": rng.uniform(*self.moire_angle_delta_range),
                    "phase": rng.uniform(*self.moire_phase_range),
                    "secondary_phase": rng.uniform(*self.moire_phase_range),
                    "grid_size": grid,
                    "hue": float(hue % 360.0),
                    "saturation": rng.uniform(*self.moire_saturation_range),
                    "value_min": float(self.moire_value_range[0]),
                    "value_max": float(self.moire_value_range[1]),
                    "hue_source": hue_source,
                    "color_mode": self.color_mode,
                })
            if self.pattern_blur_sigma and self.pattern_blur_sigma > 0:
                cfg["blur_sigma"] = float(self.pattern_blur_sigma)
            cfg["name"] = self._name(cfg)
            cats.append(cfg)
        return cats

    def _name(self, cfg: Dict) -> str:
        fam = cfg["family"]
        if fam == "solid":
            base = "solid"
        elif fam == "checker":
            base = f"checker_sz{cfg['tile']}"
        elif fam == "stripes":
            base = f"stripes_{cfg['angle']}deg_w{cfg['width']}"
        elif fam == "dots":
            base = f"dots_r{cfg['radius']}_s{cfg['spacing']}"
        elif fam == "circle":
            base = f"circle_n{cfg['count']}_r{cfg['rmin']}-{cfg['rmax']}"
        elif fam == "blob":
            base = f"blob_s{int(cfg['scale'])}_t{int(cfg['thresh']*100)}"
        elif fam == "perlin":
            base = f"perlin_s{int(cfg['scale'])}"
        elif fam == "texture":
            base = "texture"
        elif fam == "moire":
            base = (
                f"moire_f{int(round(cfg['frequency']))}"
                f"_df{int(round(cfg['frequency_delta'] * 10))}"
                f"_a{int(round(cfg['angle']))}"
            )
            if cfg.get("grid_size"):
                base += f"_grid{int(cfg['grid_size'])}"
            base += f"_h{int(round(cfg['hue']))}"
        else:
            base = "unknown"
        if "blur_sigma" in cfg and cfg["blur_sigma"] and cfg["blur_sigma"] > 0:
            base += f"_blurS{int(round(cfg['blur_sigma']))}"
        cm = (cfg.get("color_mode") or self.color_mode).lower()
        if cm in ("gray","bw","ood"):
            base += f"_{cm}"
        return base

@dataclass
class GenerateDistractor:
    """Main class for generating synthetic distractor images.
    
    Creates non-semantic pattern-based images designed to test model robustness.
    Supports various pattern families (checker, stripes, dots, etc.) with 
    configurable color modes, blur effects, and texture integration.
    
    Args:
        image_size: Output image dimensions (square images)
        spec: CategorySpec defining pattern types and parameters
        color_sampler: ColorSampler for gamut-based or random color generation
        global_blur_sigma: Defocus blur applied after pattern generation (unlabeled)
        texture_dir: Directory containing texture images for background mixing
        include_texture_family: Whether to add texture-based patterns
        texture_mix_prob: Probability of mixing textures into pattern backgrounds
        seed: Random seed for reproducible generation
    
    Example:
        >>> sampler = ColorSampler(seed=42)
        >>> spec = CategorySpec.default(families=["checker", "stripes"], n_categories=10)
        >>> gen = GenerateDistractor(spec=spec, color_sampler=sampler)
        >>> img, name = gen()  # Generate single distractor image
    """
    image_size: int = 224
    spec: CategorySpec = field(default_factory=CategorySpec)
    color_sampler: ColorSampler = field(default_factory=ColorSampler)
    global_blur_sigma: float = 0.0
    texture_dir: Optional[str] = None
    include_texture_family: bool = False
    texture_mix_prob: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.texture_paths = []
        if self.texture_dir:
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                self.texture_paths += glob.glob(os.path.join(self.texture_dir, "**", ext), recursive=True)
        if self.include_texture_family and "texture" not in self.spec.families:
            self.spec.families.append("texture")
        self.categories = self.spec.instantiate(self.rng)

    def _solid_color(self, mode):
        color = self.color_sampler.sample_color(mode)
        return Image.new("RGB", (self.image_size, self.image_size), color)

    def _maybe_texture_bg(self):
        if self.texture_paths and self.rng.random() < self.texture_mix_prob:
            p = self.rng.choice(self.texture_paths)
            try:
                im = Image.open(p).convert("RGB")
                if im.width < self.image_size or im.height < self.image_size:
                    im = im.resize((max(self.image_size, im.width), max(self.image_size, im.height)))
                x = self.rng.randint(0, im.width - self.image_size)
                y = self.rng.randint(0, im.height - self.image_size)
                return im.crop((x,y,x+self.image_size,y+self.image_size))
            except Exception:
                return None
        return None

    def _color_pair(self, mode):
        c1 = self.color_sampler.sample_color(mode)
        c2 = self.color_sampler.sample_color(mode)
        return c1, c2

    def _render_checker(self, tile, mode):
        bg = self._maybe_texture_bg() or self._solid_color(mode)
        arr = np.array(bg)
        sz = self.image_size
        yy, xx = np.mgrid[0:sz, 0:sz]
        tile = max(2, int(tile))
        pattern = (((yy // tile) + (xx // tile)) % 2) == 0
        c1, c2 = self._color_pair(mode)
        arr[pattern] = c1
        arr[~pattern] = c2
        return Image.fromarray(arr)

    def _render_stripes(self, angle, width, mode):
        bg = self._maybe_texture_bg() or self._solid_color(mode)
        arr = np.array(bg)
        sz = self.image_size
        yy, xx = np.mgrid[0:sz, 0:sz]
        theta = math.radians(angle)
        proj = (np.cos(theta)*xx + np.sin(theta)*yy)
        period = max(2, int(width*2))
        stripe = ((proj // period) % 2) == 0
        c1, c2 = self._color_pair(mode)
        arr[stripe] = c1
        arr[~stripe] = c2
        return Image.fromarray(arr)

    def _render_dots(self, radius, spacing, mode):
        img = self._maybe_texture_bg() or self._solid_color(mode)
        draw = ImageDraw.Draw(img)
        color = self.color_sampler.sample_color(mode)
        for y in range(0, self.image_size, spacing):
            for x in range(0, self.image_size, spacing):
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        return img

    def _perlin(self, scale):
        sz = self.image_size
        grid = int(max(1, sz/scale))
        coarse = self.rng.random() * np.ones((grid+2, grid+2))
        
        # Vectorized coordinate arrays (500x faster than nested loops)
        y_coords, x_coords = np.mgrid[0:sz, 0:sz]
        gx, gy = x_coords/scale, y_coords/scale
        x0, y0 = gx.astype(int), gy.astype(int)
        dx, dy = gx - x0, gy - y0
        
        # Vectorized bilinear interpolation
        c00 = coarse[y0, x0]; c10 = coarse[y0, x0+1]
        c01 = coarse[y0+1, x0]; c11 = coarse[y0+1, x0+1]
        nx0 = (1-dx)*c00 + dx*c10
        nx1 = (1-dx)*c01 + dx*c11
        img = (1-dy)*nx0 + dy*nx1
        
        img = (img - img.min()) / (img.ptp() + 1e-8)
        return img

    def _render_blob(self, scale, thresh, mode):
        noise = self._perlin(scale)
        mask = noise > thresh
        c_fg = self.color_sampler.sample_color(mode)
        c_bg = self.color_sampler.sample_color(mode)
        arr = np.where(mask[...,None], c_fg, c_bg).astype(np.uint8)
        return Image.fromarray(arr)

    def _render_circle(self, count, rmin, rmax, mode):
        img = self._maybe_texture_bg() or self._solid_color(mode)
        draw = ImageDraw.Draw(img)
        for _ in range(count):
            r = self.rng.randint(rmin, rmax)
            x = self.rng.randint(r, self.image_size-r)
            y = self.rng.randint(r, self.image_size-r)
            color = self.color_sampler.sample_color(mode)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        return img

    def _render_perlin(self, scale, mode):
        noise = self._perlin(scale)
        c1 = self.color_sampler.sample_color(mode)
        c2 = self.color_sampler.sample_color(mode)
        arr = (noise[...,None]*np.array(c1) + (1-noise[...,None])*np.array(c2)).astype(np.uint8)
        return Image.fromarray(arr)

    def _render_texture(self):
        if not self.texture_paths:
            return self._solid_color(self.spec.color_mode)
        p = self.rng.choice(self.texture_paths)
        try:
            im = Image.open(p).convert("RGB")
            if im.width < self.image_size or im.height < self.image_size:
                im = im.resize((max(self.image_size, im.width), max(self.image_size, im.height)))
            x = self.rng.randint(0, im.width - self.image_size)
            y = self.rng.randint(0, im.height - self.image_size)
            return im.crop((x,y,x+self.image_size,y+self.image_size))
        except Exception:
            return self._solid_color(self.spec.color_mode)

    def generate_category(self, cfg: Dict) -> Image.Image:
        fam = cfg["family"]
        mode = cfg.get("color_mode", self.spec.color_mode)
        if fam == "solid":
            img = self._solid_color(mode)
        elif fam == "checker":
            img = self._render_checker(cfg["tile"], mode)
        elif fam == "stripes":
            img = self._render_stripes(cfg["angle"], cfg["width"], mode)
        elif fam == "dots":
            img = self._render_dots(cfg["radius"], cfg["spacing"], mode)
        elif fam == "circle":
            img = self._render_circle(cfg["count"], cfg["rmin"], cfg["rmax"], mode)
        elif fam == "blob":
            img = self._render_blob(cfg["scale"], cfg["thresh"], mode)
        elif fam == "perlin":
            img = self._render_perlin(cfg["scale"], mode)
        elif fam == "texture":
            img = self._render_texture()
        elif fam == "moire":
            img = self._render_moire(
                frequency=cfg["frequency"],
                frequency_delta=cfg["frequency_delta"],
                angle=cfg["angle"],
                angle_delta=cfg["angle_delta"],
                phase=cfg["phase"],
                secondary_phase=cfg["secondary_phase"],
                hue=cfg["hue"],
                saturation=cfg["saturation"],
                value_min=cfg["value_min"],
                value_max=cfg["value_max"],
                grid_size=cfg.get("grid_size"),
            )
        else:
            img = self._solid_color(mode)

        if "blur_sigma" in cfg and cfg["blur_sigma"] and cfg["blur_sigma"] > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(cfg["blur_sigma"])))
        if self.global_blur_sigma and self.global_blur_sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(self.global_blur_sigma)))
        return img

    def __call__(self):
        cfg = self.rng.choice(self.categories)
        img = self.generate_category(cfg)
        return img, cfg["name"]

    def _apply_grid(self, arr: np.ndarray, grid_size: Optional[int]) -> np.ndarray:
        if not grid_size or grid_size <= 1:
            return arr
        sz = arr.shape[0]
        # Ensure integer boundaries even when sz not divisible by grid_size
        y_edges = np.linspace(0, sz, grid_size + 1, dtype=int)
        x_edges = np.linspace(0, sz, grid_size + 1, dtype=int)
        quantized = arr.copy()
        for yi in range(grid_size):
            y0, y1 = y_edges[yi], y_edges[yi + 1]
            for xi in range(grid_size):
                x0, x1 = x_edges[xi], x_edges[xi + 1]
                block = quantized[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                block_mean = float(block.mean())
                block[:] = block_mean
        return quantized

    def _render_moire(self, frequency, frequency_delta, angle, angle_delta, phase,
                      secondary_phase, hue, saturation, value_min, value_max,
                      grid_size: Optional[int]):
        sz = self.image_size
        coords = np.linspace(0.0, 1.0, sz, endpoint=False)
        xx, yy = np.meshgrid(coords, coords)
        base_theta = math.radians(angle)
        secondary_theta = math.radians(angle + angle_delta)
        base_freq = max(frequency, 0.1)
        sec_freq = max(frequency + frequency_delta, 0.1)

        proj_base = np.cos(base_theta) * xx + np.sin(base_theta) * yy
        proj_secondary = np.cos(secondary_theta) * xx + np.sin(secondary_theta) * yy

        pattern = 0.5 * (
            np.sin(2 * math.pi * base_freq * proj_base + phase)
            + np.sin(2 * math.pi * sec_freq * proj_secondary + secondary_phase)
        )
        pattern = (pattern - pattern.min()) / (pattern.ptp() + 1e-6)
        pattern = self._apply_grid(pattern, grid_size)

        sat = float(np.clip(saturation, 0.0, 1.0))
        vmin = float(np.clip(value_min, 0.0, 1.0))
        vmax = float(np.clip(value_max, 0.0, 1.0))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        values = vmin + pattern * (vmax - vmin)

        h_prime = (hue % 360.0) / 60.0
        chroma = values * sat
        x = chroma * (1 - np.abs((h_prime % 2) - 1))
        m = values - chroma

        zeros = np.zeros_like(values)
        r_prime = np.zeros_like(values)
        g_prime = np.zeros_like(values)
        b_prime = np.zeros_like(values)

        if 0 <= h_prime < 1:
            r_prime, g_prime, b_prime = chroma, x, zeros
        elif 1 <= h_prime < 2:
            r_prime, g_prime, b_prime = x, chroma, zeros
        elif 2 <= h_prime < 3:
            r_prime, g_prime, b_prime = zeros, chroma, x
        elif 3 <= h_prime < 4:
            r_prime, g_prime, b_prime = zeros, x, chroma
        elif 4 <= h_prime < 5:
            r_prime, g_prime, b_prime = x, zeros, chroma
        else:
            r_prime, g_prime, b_prime = chroma, zeros, x

        rgb = np.stack([r_prime + m, g_prime + m, b_prime + m], axis=-1)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)
