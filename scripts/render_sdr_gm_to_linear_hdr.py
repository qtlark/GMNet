import cv2
import numpy as np


sdr = cv2.imread("sdr.jpg", cv2.IMREAD_UNCHANGED).astype("float32") / 255.0

ngm = cv2.imread("ngm.jpg", cv2.IMREAD_UNCHANGED).astype("float32") / 255.0
ngm = cv2.resize(ngm, None, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)[...,None]

qmax = np.load('meta.npy')
gm = qmax*ngm

hdr = np.power(sdr, 2.2) * np.power(2, gm)
cv2.imwrite("hdr_linear.hdr", hdr)
