from model import Generator
import numpy as np
import cv2
import argparse

generator = Generator()
# generator.summary()
SIZE = 1024


def inference(input_path, output_path):
    inp = cv2.imread(input_path)
    h, w , _ = inp.shape
    inp = inp / 255.
    inp = cv2.resize(inp, (SIZE, SIZE))
    inp = np.expand_dims(inp, 0)

    out = generator(inp, training=True)[0].numpy()
    out = cv2.resize(out, (w, h))
    out = (out * 255).astype(int)
    # print(out)

    cv2.imwrite(output_path, out)
    print("save output ok!")


def infer_train_data(input_path, output_path):
    inp = cv2.imread(input_path)
    inp = cv2.resize(inp, (SIZE * 2, SIZE))
    real = inp[:, SIZE:, :]
    inp = inp[:, :SIZE, :]
    inp = inp / 255.
    inp = np.expand_dims(inp, 0)

    out = generator(inp, training=True)[0].numpy()
    print(out)
    print(np.max(out), np.min(out))
    out = (out * 255).astype(int)

    compare = np.concatenate([out, real], axis=1)
    cv2.imwrite(output_path, compare)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--weights_path', help='weights_path', default="weights/1st_save_gen_1024_w.h5")
    ap.add_argument('-i', '--input_path', help='input_path', default="tests/inp/test1.jpg")
    ap.add_argument('-o', '--output_path', help='output_path', default="tests/out/test1.jpg")
    args = ap.parse_args()

    generator.load_weights(args.weights_path)

    inference(args.input_path, args.output_path)

    # input_path = "dataset/train/10744973825.jpg"
    # output_path = "tests/out/train/10744973825.jpg"
    # infer_train_data(input_path, output_path)
