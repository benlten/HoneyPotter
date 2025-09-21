from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any, Optional, Callable
import json
import numpy as np
import random

ColorSpace = Literal["RGB", "LinearRGB", "HSV", "Lab"]
Method = Literal["hist-hdr", "gmm"]

@dataclass
class GamutSpec:
    name: str
    version: int
    color_space: ColorSpace
    reference_white: str | None
    method: Method
    payload: Dict[str, Any]

def load_gamut(path: str) -> GamutSpec:
    """Load gamut specification from JSON file."""
    with open(path, "r") as f:
        d = json.load(f)
    # Validate required keys
    required_keys = ["name", "version", "color_space", "reference_white", "method", "payload"]
    return GamutSpec(**{k: d[k] for k in required_keys})

def make_hist_hdr_checker(spec: GamutSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Create histogram-based HDR gamut checker."""
    p = spec.payload
    bins = p["bins_per_axis"]
    lohi = np.array(p["ranges"], dtype=np.float32)  # shape (3,2)
    occ = np.zeros((bins, bins, bins), dtype=bool)
    for i, j, k in p["occupied_bins"]:
        occ[i, j, k] = True
    scale = (bins / (lohi[:, 1] - lohi[:, 0])).astype(np.float32)
    shift = (-lohi[:, 0] * scale).astype(np.float32)

    def pixel_inlier(x: np.ndarray) -> np.ndarray:
        # x: (..., 3) in the spec's color space
        idx = np.floor(x * scale + shift).astype(int)
        idx = np.clip(idx, 0, bins - 1)
        return occ[idx[..., 0], idx[..., 1], idx[..., 2]]

    return pixel_inlier

def make_gmm_checker(spec: GamutSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Create GMM-based gamut checker."""
    p = spec.payload
    w = np.asarray(p["weights"], dtype=np.float64)        # (K,)
    mu = np.asarray(p["means"], dtype=np.float64)         # (K,3)
    Sigma = np.asarray(p["covariances"], dtype=np.float64)# (K,3,3)
    invS = np.linalg.inv(Sigma)
    logdet = np.log(np.linalg.det(Sigma))
    thr = float(p["loglik_threshold"])
    K = w.shape[0]
    const = -0.5 * (3 * np.log(2 * np.pi) + logdet) + np.log(w)

    def pixel_loglik(x: np.ndarray) -> np.ndarray:
        # x: (N,3)
        N = x.shape[0]
        ll = np.full((N, K), -np.inf, dtype=np.float64)
        for k in range(K):
            d = x - mu[k]
            m = np.einsum("ni,ij,nj->n", d, invS[k], d)
            ll[:, k] = const[k] - 0.5 * m
        return np.logaddexp.reduce(ll, axis=1)

    def pixel_inlier(x: np.ndarray) -> np.ndarray:
        return pixel_loglik(x) >= thr

    return pixel_inlier

def make_gamut_checker(spec: GamutSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Create appropriate gamut checker based on method."""
    if spec.method == "hist-hdr":
        return make_hist_hdr_checker(spec)
    elif spec.method == "gmm":
        return make_gmm_checker(spec)
    else:
        raise ValueError(f"Unsupported gamut method: {spec.method}")

class ColorSampler:
    """Handles color sampling with optional gamut constraints.
    
    Provides sophisticated color generation using either gamut-based sampling
    (when gamut file is provided) or simple seeded random RGB generation.
    Supports special color modes like grayscale, black/white, and OOD colors.
    
    Args:
        gamut_path: Optional path to gamut JSON file for boundary-based sampling
        seed: Random seed for reproducible color generation
        
    Example:
        >>> # Simple random RGB
        >>> sampler = ColorSampler(seed=42)
        >>> color = sampler.sample_color("color")  # (127, 89, 201)
        
        >>> # Gamut-based sampling  
        >>> sampler = ColorSampler("./imagenet_gamut.json", seed=42)
        >>> color = sampler.sample_color("color")  # Color within gamut boundaries
    """
    
    def __init__(self, gamut_path: Optional[str] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.gamut_checker = None
        
        if gamut_path:
            spec = load_gamut(gamut_path)
            self.gamut_checker = make_gamut_checker(spec)
    
    def sample_color(self, mode: str = "color", max_retries: int = 100) -> Tuple[int, int, int]:
        """Sample a color according to the specified mode."""
        if mode == "gray":
            val = self.rng.randint(0, 255)
            return (val, val, val)
        elif mode == "bw":
            return (255, 255, 255) if self.rng.random() < 0.5 else (0, 0, 0)
        elif mode == "ood":
            return self._sample_ood_color()
        else:  # mode == "color"
            if self.gamut_checker:
                return self._sample_from_gamut(max_retries)
            else:
                return self._sample_random_rgb()
    
    def _sample_random_rgb(self) -> Tuple[int, int, int]:
        """Sample random RGB color with seed."""
        return (
            self.rng.randint(0, 255),
            self.rng.randint(0, 255),
            self.rng.randint(0, 255)
        )
    
    def _sample_from_gamut(self, max_retries: int) -> Tuple[int, int, int]:
        """Sample color from gamut boundaries with batch optimization."""
        # Batch sampling for better performance
        batch_size = min(max_retries, 50)
        
        try:
            # Generate batch of candidate colors
            colors = np.random.RandomState(self.rng.getstate()[1][0]).randint(0, 256, (batch_size, 3))
            color_normed = colors.astype(np.float64) / 255.0
            
            # Check all candidates at once
            valid_mask = self.gamut_checker(color_normed)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Return first valid color
                idx = valid_indices[0]
                return tuple(colors[idx].astype(int))
        except Exception:
            # Fallback to single sampling on any error
            pass
        
        # Single sample fallback
        for _ in range(min(max_retries, 10)):
            r, g, b = self._sample_random_rgb()
            color_array = np.array([r, g, b]).reshape(1, 3) / 255.0
            
            try:
                if self.gamut_checker(color_array)[0]:
                    return (r, g, b)
            except Exception:
                continue
        
        # Final fallback
        return self._sample_random_rgb()
    
    def _sample_ood_color(self) -> Tuple[int, int, int]:
        """Sample out-of-distribution (neon) colors."""
        # Default neon palette
        ood_colors = [
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 102),    # Neon green
            (255, 102, 0),    # Neon orange
            (102, 0, 255),    # Neon purple
            (255, 255, 0),    # Yellow
        ]
        return self.rng.choice(ood_colors)

def color_from_mode(rng: random.Random, mode: str, gamut_checker: Optional[Callable] = None) -> Tuple[int, int, int]:
    """Legacy function for backward compatibility."""
    sampler = ColorSampler(seed=rng.getstate()[1][0] if rng else None)
    sampler.gamut_checker = gamut_checker
    sampler.rng = rng
    return sampler.sample_color(mode)