import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import ImageDraw

def draw_point_on_img_dict(img, in_dict, label=None, point_num = 14):
    """
    :param img: ori_img, [B, 3, H, W]
    """
    colors = (
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 127, 255),
        (255, 0, 127),
        (127, 255, 0),
        (0, 255, 127),
        (127, 0, 255),
        (255, 127, 0)
    )
    line_pair = (
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
        (10, 11),
        (11, 12),
        (13, 14),
    )

    if label is not None:
        length = len(label)
        for idx in xrange(length):
            coords = label[idx].reshape(point_num, 3).astype(np.int32)
            xx = coords[:, 0]
            yy = coords[:, 1]
            vv = coords[:, 2]
            for i in xrange(point_num):
                x = xx[i]
                y = yy[i]
                v = vv[i]
                if v == 1:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)

    idx = 0
    for key, val in in_dict.items():
        coords = np.array(val).reshape(point_num, 3).astype(np.int32)
        xx = coords[:, 0]
        yy = coords[:, 1]
        vv = coords[:, 2]
        for i in xrange(point_num):
            x = xx[i]
            y = yy[i]
            v = vv[i]
            if v == 1:
                cv2.circle(img, (x, y), 2, colors[idx], 2)
        for i1, i2 in line_pair:
            i1 -= 1
            i2 -= 1

            x1 = xx[i1]
            y1 = yy[i1]
            v1 = vv[i1]

            x2 = xx[i2]
            y2 = yy[i2]
            v2 = vv[i2]

            if v1 == 1 and v2 == 1:
                cv2.line(img, (x1, y1), (x2, y2), colors[idx])
        if idx < len(colors):
            idx += 1
        else:
            idx = len(colors) - 1

    plt.imshow(img)
    plt.show()

