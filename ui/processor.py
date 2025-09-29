# processor.py
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch

MEAN, STD = 0.1307, 0.3081

def _bg_from_corners(gray_arr: np.ndarray) -> int:
    h, w = gray_arr.shape
    corners = np.r_[gray_arr[:10,:10].ravel(), gray_arr[:10,-10:].ravel(),
                    gray_arr[-10:,:10].ravel(), gray_arr[-10:,-10:].ravel()]
    return int(np.median(corners))

def preprocess(pil_img: Image.Image) -> torch.Tensor | None:
    g = ImageOps.grayscale(pil_img)
    a = np.asarray(g).astype(np.uint8)
    if _bg_from_corners(a) > 127:
        g = ImageOps.invert(g); a = 255 - a
    thr = max(50, min(200, _bg_from_corners(a) + 20))
    mask = (a > thr).astype(np.uint8)
    if mask.sum() < 30:
        return None
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    crop = g.crop((x0, y0, x1, y1)).filter(ImageFilter.MaxFilter(3))
    crop.thumbnail((20, 20), Image.LANCZOS)
    canvas28 = Image.new("L", (28, 28), 0)
    ox, oy = (28 - crop.size[0]) // 2, (28 - crop.size[1]) // 2
    canvas28.paste(crop, (ox, oy))
    arr = np.asarray(canvas28).astype("float32") / 255.0
    t = torch.from_numpy(arr)[None, None, ...]         # [1,1,28,28]
    t = (t - MEAN) / STD
    return t
