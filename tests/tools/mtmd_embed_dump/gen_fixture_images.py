#!/usr/bin/env python3
"""Generate the deterministic vision fixture PNGs.

Each image: a diagonal RGB gradient, three filled rectangles and one
diagonal line, so no axis flip or transpose maps the
image onto itself. Pure integer arithmetic, no random source.

Usage: python3 gen_fixture_images.py <out_dir>
"""
import sys
import numpy as np
from PIL import Image


def make(w, h):
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    r = np.broadcast_to((x * 255) // max(w - 1, 1), (h, w))
    g = np.broadcast_to((y * 255) // max(h - 1, 1), (h, w))
    b = ((x + y) * 255) // max(w + h - 2, 1)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    # rectangles: (x0, y0, x1, y1, color)
    rects = [
        (w // 8, h // 8, w * 3 // 8, h * 3 // 8, (255, 32, 32)),
        (w // 2, h // 5, w * 7 // 8, h // 2, (32, 255, 32)),
        (w // 3, h * 5 // 8, w * 2 // 3, h * 7 // 8, (32, 32, 255)),
    ]
    for x0, y0, x1, y1, c in rects:
        img[y0:y1, x0:x1] = c
    # diagonal line from top-left to bottom-right, 2px wide
    for yy in range(h):
        xx = (yy * (w - 1)) // max(h - 1, 1)
        img[yy, max(xx - 1, 0):min(xx + 1, w)] = (255, 255, 255)
    return img


def main():
    out = sys.argv[1]
    for w, h in [(96, 64), (400, 300)]:
        Image.fromarray(make(w, h), "RGB").save(f"{out}/img_{w}x{h}.png", optimize=True)


if __name__ == "__main__":
    main()
