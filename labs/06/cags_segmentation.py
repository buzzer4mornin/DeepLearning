#!/usr/bin/env python3

# Team:
# 3351ff04-3f62-11e9-b0fd-00505601122b
# ff29975d-0276-11eb-9574-ea7484399335

# Team members: Aydin Ahmadli, Filip Jurčák


import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

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
    cags = CAGS()

    def with_mask(x):
        return x["image"], x["mask"]

    train = cags.train
    train = train.map(with_mask)
    train = train.shuffle(5000, seed=args.seed)
    train = train.batch(args.batch_size)

    dev = cags.dev
    dev = dev.map(with_mask)
    dev = dev.batch(args.batch_size)

    test = cags.test
    test = test.map(with_mask)
    test = test.batch(args.batch_size)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = False

    inputs = tf.keras.layers.Input([cags.H, cags.W, cags.C])

    # Lets use all outputs, except first one by networks. We'll use those which are intermediate results of the network
    # Not the final efficientnet output
    features = efficientnet_b0(inputs)

    # Lets use features for reconstruction of segmentation.
    f = features[1]
    f = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', use_bias=False)(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.ReLU()(f)

    for feature in features[2:]:
        f = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False)(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        f = tf.keras.layers.Dropout(rate=0.3)(f)

        f = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        f = tf.keras.layers.Dropout(rate=0.2)(f)

        f = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        f = tf.keras.layers.Dropout(rate=0.1)(f)

        f_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', use_bias=False)(feature)
        f_1 = tf.keras.layers.BatchNormalization()(f_1)
        f_1 = tf.keras.layers.ReLU()(f_1)
        f = tf.keras.layers.Dropout(rate=0.3)(f)
        f = tf.keras.layers.Add()([f, f_1])

    # Add last layer with transposed conv + sigmoid activation for having pixels in range [0,1]
    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same',
                                              activation="sigmoid")(f)

    # TODO: Create the model and train it
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()

    # Custom PiecewiseConstantDecay - half the learning rate at each half
    def lr_schedule(epoch):
        if epoch > args.epochs * 3 / 4:
            lr = 0.00001
        elif epoch > args.epochs / 2:
            lr = 0.0001
        else:
            lr = 0.001
        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0)),
        loss=tf.losses.Huber(),
        metrics=[cags.MaskIoUMetric()]
    )

    # Fit the model
    model.fit(
        train,
        shuffle=True,
        epochs=args.epochs,
        callbacks=[lr_scheduler],
        validation_data=dev, verbose=2
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
