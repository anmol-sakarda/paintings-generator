from PIL import Image
import os
from tqdm.notebook import tqdm
import numpy as np
from keras.layers import Dense, Reshape, Flatten, Input, BatchNormalization, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt


def load_data(path, img_dim):
    images = []
    for file in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, file))
        img = img.resize((img_dim, img_dim))
        img = np.array(img)
        img = (img - 127.5) / 127.5
        images.append(img)
    return np.asarray(images)


data_dir = 'image_folder/Images'
img_dim = 128
images = load_data(data_dir, img_dim)


def build_generator():
    net = Sequential()
    net.add(Dense(16 * 16 * 256, input_dim=100))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('relu'))
    net.add(Reshape((16, 16, 256)))
    net.add(Dropout(0.4))

    net.add(UpSampling2D())
    net.add(Conv2D(128, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('relu'))

    net.add(UpSampling2D())
    net.add(Conv2D(128, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('relu'))

    net.add(UpSampling2D())
    net.add(Conv2D(64, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('relu'))

    net.add(Conv2D(32, 5, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('relu'))

    net.add(Conv2D(3, 5, padding='same'))
    net.add(Activation('tanh'))

    # net.summary()

    return net


def build_discriminator():
    net = Sequential()
    net.add(Conv2D(64, 5, strides=2, input_shape=(128, 128, 3), padding='same'))
    net.add(LeakyReLU())

    net.add(Conv2D(128, 5, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(0.4))

    net.add(Conv2D(256, 5, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(0.4))

    net.add(Conv2D(512, 5, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(0.4))

    net.add(Flatten())
    net.add(Dense(1))
    net.add(Activation('sigmoid'))

    # net.summary()

    return net


discriminator_model = build_discriminator()
discriminator_optimizer = RMSprop(lr=0.0002, clipvalue=1.0, decay=6e-8)
discriminator_model.compile(loss='binary_crossentropy',
                            optimizer=discriminator_optimizer,
                            metrics=['accuracy'])

adversarial_model = Sequential()
generator = build_generator()
adversarial_model.add(generator)

# discriminator layers frozen so only generator layers will train
for layer in discriminator_model.layers:
    layer.trainable = False

adversarial_model.add(discriminator_model)
adversarial_optimizer = Adam(lr=0.0001, clipvalue=1.0, decay=3e-8)
adversarial_model.compile(loss='binary_crossentropy',
                          optimizer=adversarial_optimizer,
                          metrics=['accuracy'])


def train(epoch, batch_size):
    np.random.shuffle(images)
    total_steps = images.shape[0] // batch_size

    for step in tqdm(range(total_steps)):
        # generate fake images by passing random noise into generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        images_fake = generator.predict(noise)

        # combine fake and real images
        images_real = images[step * batch_size: (step + 1) * batch_size]
        x = np.concatenate((images_fake, images_real))

        # create training labels
        y = np.zeros([2 * batch_size, 1])
        y[batch_size:, :] = 1

        # train discriminator on fake and real images
        d_stats = discriminator_model.train_on_batch(x, y)

        # train generator based on ability to fool discriminator
        noise = np.random.normal(0, 1, (batch_size * 2, 100))
        a_stats = adversarial_model.train_on_batch(noise, np.ones([batch_size * 2, 1]))

    print(d_stats)
    print(a_stats)
    print('---------------------------------------------------------')


vis_noise = np.random.normal(0, 1, (16, 100))


def genSample(path, epoch):
    generated_images = generator.predict(vis_noise)
    plt.figure(figsize=(10, 10))
    generated_images = (generated_images * 0.5) + 0.5

    for im in range(generated_images.shape[0]):
        plt.subplot(4, 4, im + 1)
        image = generated_images[im, :, :, :]
        plt.imshow(image)
        plt.axis('off')

    plot = path + '/Epoch{}.png'.format(epoch)
    plt.savefig(plot)
    plt.close('all')


batch_size = 128

training_sample_path = 'image_folder/Images'
for epoch in range(1, 1001):
    genSample(training_sample_path, epoch)
    train(epoch, batch_size)
