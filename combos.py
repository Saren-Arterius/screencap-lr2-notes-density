#!/usr/bin/env python3
from math import ceil
import cv2
import numpy
from PIL import Image
import pyscreenshot as ImageGrab
from time import sleep, time

method = cv2.TM_SQDIFF_NORMED
thresold = 0.5
density_thresold = 40

def split_to(im, rows, columns, show=False):
    results = []
    for r in range(rows):
        rs = []
        for c in range(columns):
            box = (
                (c * im.size[0]) / columns,
                (r * im.size[1]) / rows,
                ((c + 1) * im.size[0]) / columns,
                ((r + 1) * im.size[1]) / rows
            )
            if show:
                im.crop(box).show()
            rs.append(cv2.cvtColor(numpy.array(
                im.crop(box)), cv2.COLOR_RGB2BGR))
        results.append(rs)
    return results

digits = split_to(Image.open("digits.png"), 3, 10)
ts = split_to(Image.open("ts.png"), 3, 1)


def get_combos():
    im = ImageGrab.grab()

    im.thumbnail((im.size[0] / 2.5, im.size[1] / 2.5))
    im = im.crop((0, im.size[1] / 2, im.size[0], im.size[1]))
    large_image = cv2.cvtColor(numpy.array(im), cv2.COLOR_RGB2BGR)

    shining_types = []
    xys = []
    trows, tcols = ts[0][0][:2]
    for shining_type in range(len(ts)):
        t_image = ts[shining_type][0]
        result = cv2.matchTemplate(t_image, large_image, method)
        mn, _, mnLoc, _ = cv2.minMaxLoc(result)
        # print(mn)
        shining_types.append(mn)
        xys.append(mnLoc)
    shining_type = min(range(len(shining_types)),
                       key=shining_types.__getitem__)
    # print(shining_type, xys[shining_type])
    x1, y1 = xys[shining_type]

    # Hard code
    # im.show()
    # im.crop().show()
    combos_field = split_to(im.crop((x1 + 85, y1, x1 + 165, y1 + 40)), 1, 4)

    combos_arr = []
    for cd_im in combos_field[0]:
        mins = []
        for md, md_im in enumerate(digits[shining_type]):
            start = time()
            large_image = cd_im
            result = cv2.matchTemplate(md_im, large_image, method)
            mn, _, mnLoc, _ = cv2.minMaxLoc(result)
            mins.append(mn)
        if min(mins) > thresold:
            break
        digit = min(range(len(mins)), key=mins.__getitem__)
        combos_arr.append(str(digit))

    combos = int("".join(combos_arr)) if len(combos_arr) else None
    return combos

if __name__ == "__main__":
    # compensate = 0
    last_combos = 0
    lag = 1
    while True:
        start = time()
        combos = get_combos()
        duration = time() - start
        if lag > 10:
            last_combos = 0
        if not combos:
            lag += duration
            continue
        density = (combos - last_combos) / (duration if duration > 1 else lag)
        if not combos or combos < last_combos or density > density_thresold:
            lag += duration
            continue
        print("Notes: {0}, Density: {1} notes/s, {2}s, {3}s".format(combos,
              round(density, 2), round(lag, 2), round(duration, 2)))
        lag = 1
        last_combos = combos
        if duration <= 1:
            sleep(1 - duration)
