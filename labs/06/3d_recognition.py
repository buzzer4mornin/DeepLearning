#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# For Running on Kaggle GPU
sys.path.insert(1, '../input/modelnet-dataset/')

import numpy as np
import tensorflow as tf
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args):
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

    # Load the data
    modelnet = ModelNet(args.modelnet)
    train_images = modelnet.train.data["voxels"]
    train_labels = modelnet.train.data["labels"]
    dev_images = modelnet.dev.data["voxels"]
    dev_labels = modelnet.dev.data["labels"]
    test = modelnet.test.data["voxels"]

    # set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    def lr_schedule(e):
        if e > args.epochs * 3 / 4:
            lr = 0.0003
        elif e > args.epochs / 2:
            lr = 0.0005
        elif e > args.epochs / 4:
            lr = 0.0007
        else:
            lr = 0.001
        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=6,
                                                      min_lr=1e-5)

    # TODO: Create the model and train it D, H, W, C
    input_layer = tf.keras.layers.Input([modelnet.D, modelnet.H, modelnet.W, modelnet.C])

    l2_rate = 1e-4
    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding="same", use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(input_layer)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer1 = Activation("relu")(conv_layer1)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding="same", use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer2 = Activation("relu")(conv_layer2)

    # pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(conv_layer2)
    conv_layer3 = BatchNormalization()(conv_layer3)
    conv_layer3 = Activation("relu")(conv_layer3)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(conv_layer3)
    conv_layer4 = BatchNormalization()(conv_layer4)
    conv_layer4 = Activation("relu")(conv_layer4)

    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    # pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)
    dense_layer2 = Dense(units=512)(flatten_layer)
    dense_layer2 = BatchNormalization()(dense_layer2)
    dense_layer2 = Activation("relu")(dense_layer2)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=10, activation='softmax')(dense_layer2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(
        train_images, train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        epochs=args.epochs,
        validation_data=(dev_images, dev_labels),
        callbacks=[lr_reducer, lr_scheduler], verbose=2
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    # os.makedirs(args.logdir, exist_ok=True)
    with open("3d_recognition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
