#!/usr/bin/env python3

# Team:
# 3351ff04-3f62-11e9-b0fd-00505601122b
# ff29975d-0276-11eb-9574-ea7484399335

# Team members: Aydin Ahmadli, Filip Jurčák

import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--verbose", default=False, action="store_true", help="TF logging verbose")


def main(args):
    # ----->>> Note that script was run on Google Collab as following;
    # with tf.device('/device:GPU:0'):

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # IMAGE AUGMENTATION SETUP
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                                                      height_shift_range=0.1, horizontal_flip=True)

    # RENAME DATASETS FOR CONVENIENCE
    train_images = cifar.train.data["images"]
    train_labels = cifar.train.data["labels"]
    dev_images = cifar.dev.data["images"]
    dev_labels = cifar.dev.data["labels"]
    test_images = cifar.test.data["images"]
    train_images = train_images.astype('float32')
    dev_images = dev_images.astype('float32')
    test_images = test_images.astype('float32')

    # STANDARDIZE THE DATA
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))

    train_images = (train_images - mean) / (std + 1e-7)
    dev_images = (dev_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- ENSEMBLE MODEL ------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    '''models = []

    for model_x in range(args.models):
        np.random.seed(args.seed + model_x)
        tf.random.set_seed(args.seed + model_x)

        # Create logdir name
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in
                 sorted(vars(args).items())))
        ))

        dropout_rate = 0.2
        l2_rate = 1e-4

        # CREATE A MODEL
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                                         input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(dropout_rate + 0.1))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(dropout_rate + 0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(dropout_rate + 0.3))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # APPEND CREATED MODEL INTO LIST
        models.append(model)

        # COMPILE  MODEL INSIDE LIST
        models[-1].compile(
            optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        # SET TensorBoard
        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100,
                                                     profile_batch=0)

        # FIT MODEL INSIDE LIST
        models[-1].fit(
            train_generator.flow(tf.reshape(train_images, [-1, 32, 32, 3]), train_labels, seed=args.seed,
                                 batch_size=args.batch_size), shuffle=False,
            epochs=args.epochs, steps_per_epoch=train_images.shape[0] // args.batch_size,
            validation_data=(dev_images, dev_labels),
            callbacks=[tb_callback], verbose=2
        )

        # SAVE MODEL
        models[-1].save('cifar_' + str(model_x) + '.h5')'''

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- TEST SAVED MODELS ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # LOAD SAVED MODELS INTO LIST
    cifar_0 = tf.keras.models.load_model('cifar_0.h5')
    cifar_1 = tf.keras.models.load_model('cifar_1.h5')
    cifar_2 = tf.keras.models.load_model('cifar_2.h5')
    pred_models = [cifar_0, cifar_1, cifar_2]

    # ENSEMBLE THE PREDICTIONS
    y_list = [pred_models[m].predict(test_images, batch_size=64) for m in range(args.models)]
    y_list = sum(y_list) / len(y_list)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("cifar_competition_test.txt", "w", encoding="utf-8") as predictions_file:
        for probs in y_list:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
