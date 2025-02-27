import numpy as np


def grey_from_label(label):
    assert label >= 0 and label <= 5
    return 100 + 30 * label


def label_from_grey(gray):
    label = (gray - 100) // 30
    assert label >= 0 and label <= 5
    return label


def bbox_from_mask(mask) -> tuple[int, int, int, int]:
    mask_rows = np.any(mask, axis=1)
    mask_cols = np.any(mask, axis=0)

    def line_start_size(values):
        start = values.argmax()
        end = values.shape[0] - np.flip(values).argmax()
        return (int(start), int(end - start))

    start_x, size_x = line_start_size(mask_cols)
    start_y, size_y = line_start_size(mask_rows)
    return (start_x, start_y, size_x, size_y)


def plot_bbox(ax, bbox):
    x0 = bbox[0]
    x1 = bbox[0] + bbox[2]
    y0 = bbox[1]
    y1 = bbox[1] + bbox[3]
    ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], "-")
