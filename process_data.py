import cv2
import numpy as np
import os

REAL_DIR = "real_photos"
SKETCH_DIR = "sketch_photos"

def photo2sketch(real_path, sketch_path=None):
    img_gray = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)

    img_gray_inv = 255 - img_gray

    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                              sigmaX=0, sigmaY=0)

    def dodgeV2(image, mask):
      return cv2.divide(image, 255-mask, scale=256)

    img_blend = dodgeV2(img_gray, img_blur)
    if sketch_path == None:
        sketch_path = os.path.join(SKETCH_DIR, os.path.basename(real_path))
    cv2.imwrite(sketch_path, img_blend)

    print(f"p2sk {os.path.basename(real_path)} DONE!")


def process_data(size, real_dir, sketch_dir):
    train_dir = f"train_{size}"
    val_dir = f"val_{size}"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    imgs = os.listdir(real_dir)
    list_train_val = os.listdir(train_dir) + os.listdir(val_dir)
    print(list_train_val)
    print(imgs)
    if list_train_val == imgs:
        return "OK!"

    num_data = len(os.listdir(real_dir))
    num_val = 400
    num_train = num_data - num_val

    imgs = os.listdir(real_dir)
    for img in imgs[:num_train]:
        real_path = os.path.join(real_dir, img)
        sketch_path = os.path.join(sketch_dir, img)
        real_tensor = cv2.resize(cv2.imread(real_path), (size, size))
        sketch_tensor = cv2.resize(cv2.imread(sketch_path), (size, size))
        train_tensor = np.concatenate([sketch_tensor, real_tensor], axis=1)
        train_path = os.path.join(train_dir, img)
        cv2.imwrite(train_path, train_tensor)
    print("Build train_dir DONE!")

    for img in imgs[num_train:]:
        real_path = os.path.join(real_dir, img)
        sketch_path = os.path.join(sketch_dir, img)
        real_tensor = cv2.resize(cv2.imread(real_path), (size, size))
        sketch_tensor = cv2.resize(cv2.imread(sketch_path), (size, size))
        val_tensor = np.concatenate([sketch_tensor, real_tensor], axis=1)
        val_path = os.path.join(val_dir, img)
        cv2.imwrite(val_path, val_tensor)
    print("Build val_dir DONE!")



if __name__ == "__main__":
    list_real = os.listdir(REAL_DIR)
    list_sketch = os.listdir(SKETCH_DIR)
    if list_real != list_sketch:
        for real in list_real:
            if real not in list_sketch:
                real_path = os.path.join(REAL_DIR, real)
                photo2sketch(real_path)

    process_data(1024, REAL_DIR, SKETCH_DIR)