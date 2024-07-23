import tensorflow as tf
import os
from model import Generator, Discriminator, SIZE
from loss import generator_loss, discriminator_loss
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tqdm
from sklearn.utils import shuffle
import argparse


generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = cv2.imread(image_file)
    image = cv2.resize(image, (SIZE*2, SIZE))
    image = image / 255.

    input_image = image[:, :SIZE, :]
    real_image = image[:, SIZE:, :]

    return input_image, real_image


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        img = display_list[i]
        # img = img.transpose((2,1,0))
        img = np.array([img[:,:,2], img[:,:,1], img[:,:,0]])
        img = img.transpose((1,2,0))
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def train(epochs, batch_size, pretrained_gen=None, pretrained_disc=None, save_gen=None, save_disc=None):
    if pretrained_gen and pretrained_disc:
        generator.load_weights(pretrained_gen)
        discriminator.load_weights(pretrained_disc)
    for i in range(epochs):
        epoch = i + 1
        loss_gen = 0
        loss_disc = 0

        list_train = os.listdir("dataset/train")
        list_val = os.listdir("dataset/val")
        steps_train = len(os.listdir("dataset/train")) // batch_size
        steps_val = len(os.listdir("dataset/val")) // batch_size

        list_train = shuffle(list_train)
        pbar = tqdm.tqdm(range(steps_train))
        for j in pbar:
            data_list_photos = list_train[batch_size * j: batch_size * (j + 1)]
            list_couple = [load(os.path.join("dataset/train", image_file)) for image_file in data_list_photos]
            inputs = np.array([couple[0] for couple in list_couple])
            reals = np.array([couple[1] for couple in list_couple])

            '''train gen model'''
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # gen, disc
                gen_outputs = generator(inputs, training=True)
                disc_gen_output = discriminator([inputs, gen_outputs], training=True)
                disc_real_output = discriminator([inputs, reals], training=True)

                # Loss gen, disc
                loss_gen, _, _ = generator_loss(disc_gen_output, gen_outputs, reals)
                loss_disc = discriminator_loss(disc_real_output, disc_gen_output)
                pbar.set_description("Epoch: {0} || g_loss: {1} || d_loss: {2}".format(epoch, loss_gen, loss_disc))

                gen_gradients = gen_tape.gradient(loss_gen, generator.trainable_weights)
                generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_weights))

                disc_gradients = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
                discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_weights))

        generator.save_weights(save_gen)
        discriminator.save_weights(save_disc)

        generate_images(generator, inputs, reals)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', help='epochs', default=40)
    ap.add_argument('-b', '--batch_size', help='batch_size', default=1)
    ap.add_argument('-pg', '--pretrained_gen', help='pretrained_gen', default=None)
    ap.add_argument('-pd', '--pretrained_disc', help='pretrained_disc', default=None)
    ap.add_argument('-sg', '--save_gen', help='save_gen', default="weights/1st_save_gen_1024_w.h5")
    ap.add_argument('-sd', '--save_disc', help='save_disc', default="weights/1st_save_disc_1024_w.h5")
    args = ap.parse_args()

    train(epochs=args.epochs,
          batch_size=args.batch_size,
          pretrained_gen=args.pretrained_gen,
          pretrained_disc=args.pretrained_disc,
          save_gen=args.save_gen,
          save_disc=args.save_disc)