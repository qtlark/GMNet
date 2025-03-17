import os
import os.path as osp
import numpy as np



raw_pt = 'hdr_double_layer.jpg'
hdr_exif = os.popen("exiftool %s"%osp.join(raw_pt))
context = hdr_exif.read()
for line in context.splitlines():
    if "MP Image Start" in line:
        start_position = int(line[34:])
    elif "MP Image Length" in line:
        gm_length = int(line[34:])
hdr_exif.close()


with open(raw_pt, 'rb') as file:
    sdr_bin = file.read(start_position)
    gm_bin = file.read(gm_length)

output_path = 'ngm.jpg'
with open(output_path, 'wb') as img_file:
    img_file.write(gm_bin)

output_path = 'sdr.jpg'
with open(output_path, 'wb') as img_file:
    img_file.write(sdr_bin)



ngm_exif = os.popen("exiftool %s"%osp.join('ngm.jpg'))
context = ngm_exif.read()
for line in context.splitlines():
    if "Gain Map Max" in line:
        qmax = np.array(float(line[34:])).astype("float32")
ngm_exif.close()

np.save('meta.npy', qmax)