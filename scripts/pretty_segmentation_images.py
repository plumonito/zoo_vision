import numpy as np
from pathlib import Path
import cv2
import enlighten


def main():
    pbarm = enlighten.get_manager()

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    ALPHA = 0.5

    output_root = Path("/home/dherrera/temp")
    root = Path("/home/dherrera/data/elephants/training_data/zag_elp_cam_019")
    pattern = "ZAG-ELP-CAM-019-26.01.2025-100859-140859_0008*.jpg"
    files = list(root.glob(pattern))
    pbar = pbarm.counter(total=len(files), unit="File")
    for img_file in files:
        seg_file = Path(str(img_file).replace("_img.jpg", "_seg.png"))

        im = cv2.imread(img_file)
        seg = cv2.imread(seg_file, flags=cv2.IMREAD_GRAYSCALE)

        labels = set(np.unique(seg)) - {0}
        for i, label in enumerate(labels):
            mask = seg == label
            color = COLORS[i]

            for c in range(3):
                imc = im[:, :, c]
                imc[mask] = imc[mask] * (1 - ALPHA) + color[c] * ALPHA
                im[:, :, c] = imc
        cv2.imwrite(output_root / img_file.name, im)
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    main()
