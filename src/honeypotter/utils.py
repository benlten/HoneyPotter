import json, random, numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from PIL import Image

def hsv_to_rgb(h, s, v):
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    return int(r*255), int(g*255), int(b*255)

def rgb_gray(r,g,b):
    y = int(0.2126*r + 0.7152*g + 0.0722*b)
    return (y,y,y)

def hex_to_rgb(hx: str):
    hx = hx.strip()
    if hx.startswith("#"): hx = hx[1:]
    if len(hx) == 3: hx = "".join([c*2 for c in hx])
    return tuple(int(hx[i:i+2], 16) for i in (0,2,4))

@dataclass
class Palette:
    h_bins: list = field(default_factory=lambda: [1/6]*6)
    s_bins: list = field(default_factory=lambda: [0.2,0.3,0.3,0.2])
    v_bins: list = field(default_factory=lambda: [0.2,0.4,0.3,0.1])
    h_edges: list = field(default_factory=lambda: [0,1/6,2/6,3/6,4/6,5/6,1])
    s_edges: list = field(default_factory=lambda: [0,0.33,0.66,0.85,1.0])
    v_edges: list = field(default_factory=lambda: [0,0.25,0.5,0.75,1.0])

    @staticmethod
    def imagenet_default():
        return Palette(h_bins=[1/6]*6, s_bins=[0.15,0.35,0.35,0.15], v_bins=[0.15,0.45,0.30,0.10])

    @staticmethod
    def things_default():
        return Palette(h_bins=[1/6]*6, s_bins=[0.10,0.35,0.40,0.15], v_bins=[0.10,0.35,0.40,0.15])

    def _sample_bin(self, weights, u):
        cum=0.0
        for i,w in enumerate(weights):
            cum+=w
            if u <= cum + 1e-12: return i
        return len(weights)-1

    def sample_hsv(self, rng: Optional[random.Random]=None):
        rng = rng or random
        import numpy as np
        h_bin = self._sample_bin(self.h_bins, rng.random())
        h = float(np.random.uniform(self.h_edges[h_bin], self.h_edges[h_bin+1]))
        s_bin = self._sample_bin(self.s_bins, rng.random())
        s = float(np.random.uniform(self.s_edges[s_bin], self.s_edges[s_bin+1]))
        v_bin = self._sample_bin(self.v_bins, rng.random())
        v = float(np.random.uniform(self.v_edges[v_bin], self.v_edges[v_bin+1]))
        return h,s,v

def save_palette(palette: Palette, path: str):
    with open(path, "w") as f: json.dump(palette.__dict__, f, indent=2)

def load_palette(path: str) -> Palette:
    data = json.load(open(path,"r"))
    return Palette(**data)

DEFAULT_OOD_HEX = ["#00FFFF","#FF00FF","#00FF66","#FF0066","#39FF14","#FFD300",
                   "#7DF9FF","#FF4F00","#FE019A","#32CD32","#FF1493","#9400D3"]

class OODColorSampler:
    def __init__(self, rng: Optional[random.Random]=None, palette_json: Optional[str]=None):
        self.rng = rng or random.Random()
        self.colors = []
        if palette_json:
            data = json.load(open(palette_json,"r"))
            if "rgb_hex" in data:
                self.colors = [hex_to_rgb(h) for h in data["rgb_hex"]]
            elif "hsv" in data:
                self.colors = [tuple(int(c*255) for c in hsv_to_rgb(*trip)) for trip in data["hsv"]]
        if not self.colors:
            self.colors = [hex_to_rgb(h) for h in DEFAULT_OOD_HEX]

    def sample(self):
        return self.rng.choice(self.colors)

def color_from_mode(palette: Palette, rng: random.Random, mode: str, ood_sampler: Optional[OODColorSampler]=None):
    mode = (mode or "palette").lower()
    if mode == "bw":
        return (0,0,0) if rng.random() < 0.5 else (255,255,255)
    elif mode == "gray":
        h,s,v = palette.sample_hsv(rng); r,g,b = hsv_to_rgb(h,s,v); return rgb_gray(r,g,b)
    elif mode == "ood":
        sampler = ood_sampler or OODColorSampler(rng=rng); return sampler.sample()
    else:
        h,s,v = palette.sample_hsv(rng); return hsv_to_rgb(h,s,v)
