#!/usr/bin/env python3

# Team:
# 3351ff04-3f62-11e9-b0fd-00505601122b
# ff29975d-0276-11eb-9574-ea7484399335

# Team members: Aydin Ahmadli, Filip Jurčák

import argparse
import datetime
import os
import re
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

# ACTIVATE ONLY FOR KAGGLE NOTEBOOK
# sys.path.insert(1, '../input/efficient-net/')
# sys.path.insert(1, '../input/cagsdataset/')

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
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
    cags = CAGS()

    # Load the EfficientNet-B0 model
    # Return with image features, not with classification layer
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    def without_mask(x):
        return x["image"], x["label"]

    train = cags.train
    train = train.map(without_mask)

    a = []
    [a.append(row[1]) for row in train]
    # output classification size
    output_size = max(a).numpy() + 1
    # length of training data - 2142 rows
    set_size = len(a)

    train = train.shuffle(set_size, seed=args.seed)
    train = train.batch(args.batch_size)

    dev = cags.dev
    dev = dev.map(without_mask)
    dev = dev.batch(args.batch_size)

    test = cags.test
    test = test.map(without_mask)
    test = test.batch(args.batch_size)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Adding last additional dense layer
    inter_input = tf.keras.layers.Dense(5000, activation='relu')(efficientnet_b0.outputs[0])
    output = tf.keras.layers.Dense(output_size, activation='softmax')(inter_input)

    # TODO: Create the model and train it
    model = tf.keras.Model(inputs=efficientnet_b0.inputs, outputs=output)
    # model.summary()

    # Define a learning rate decay
    initial_lr = 0.0001
    final_lr = 0.00000001
    lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_lr, args.epochs * set_size / args.batch_size, final_lr)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(
        train,
        shuffle=True,
        epochs=args.epochs,
        validation_data=dev, verbose=2
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
