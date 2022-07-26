import numpy as np
import tensorflow as tf
import os
from model import Generator, Discriminator
from matplotlib import pyplot as plt
import cv2


generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
SIZE = 1024

# Define the optimizers and a checkpoint-saver
checkpoint_dir = f'training_checkpoints_{SIZE}'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore("training_checkpoints_1024/ckpt-8")


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def gen_image(inp_path, out_path):
    inp_tensor = cv2.imread(inp_path)
    height, width, _ = inp_tensor.shape
    inp_tensor = cv2.resize(inp_tensor, (SIZE, SIZE))
    inp_tensor = (inp_tensor/127.5) - 1
    inps = np.array([inp_tensor])
    prediction = generator(inps, training=True)[0]
    prediction = prediction * 0.5 + 0.5
    prediction = np.array(prediction)
    plt.imsave(out_path, prediction, cmap='gray')
    out_tensor = cv2.imread(out_path)
    out_tensor = cv2.resize(out_tensor, (width, height))
    cv2.imwrite(out_path, out_tensor)


if __name__ == "__main__":
    inp_path = "inp_out/inp/sketch3.jpg"
    out_path = "inp_out/out/gen3.jpg"
    gen_image(inp_path, out_path)
