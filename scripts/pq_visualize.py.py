import cv2
import numpy as np


def pq_oetf(x):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    Ym = np.power(x/10000, m1)
    return np.power((c1 + c2 * Ym) / (1.0 + c3 * Ym), m2)



img = cv2.imread("hdr_linear.hdr", cv2.IMREAD_UNCHANGED)     # linear hdr [0,peak]  e.g real_world [0, 5]
white_level = 203           # SDR white level
img = img*white_level       # to nits [0,1015]
img = pq_oetf(img)          # pq encoded hdr [0,1]
cv2.imwrite("hdr_pq.png", (img*65535).astype('uint16'))