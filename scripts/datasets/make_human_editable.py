import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import enlighten
import gc
from typing import Any


TRAINING_DATA_ROOT = Path("/home/dherrera/data/elephants/training_data")
OUTPUT_ROOT = Path("/home/dherrera/data/elephants/human_editable")

pbar_manager = enlighten.get_manager()


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = list(TRAINING_DATA_ROOT.glob("**/*_img.jpg"))
    pbar = pbar_manager.counter(total=len(files), unit="file")
    for image_file in pbar(files):
        seg_file = Path(str(image_file).replace("_img.jpg", "_seg.png"))

        im_color = cv2.imread(image_file)
        im_seg = cv2.imread(seg_file, flags=cv2.IMREAD_GRAYSCALE)

        out_path = OUTPUT_ROOT / image_file.name.replace(".jpg", ".png")
        im_color[:, :, 1] = im_seg
        cv2.imwrite(out_path, im_color)


if __name__ == "__main__":
    main()
