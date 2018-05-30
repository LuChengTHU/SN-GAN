import imageio
import glob
import numpy as np

img_list = glob.glob("/home/Data/Clever/JPEG/train/*.jpg")
num = len(img_list)
data = np.zeros((num, 320, 480, 3), dtype=np.uint8)
for idx, n in enumerate(img_list):
    print(idx)
    img = imageio.imread(n)
    data[idx] = img
    if idx % num == 0 and idx > 1:
        print(idx, data.dtype)

np.save("train_{}.npy".format(num), data.astype(np.uint8))

