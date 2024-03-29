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

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # HELPER FUNCTION FOR ADDING EACH LAYER
        def add_layer(layer, input_to_layer):
            # Convolutional Layer
            if layer.startswith("C"):
                params = layer.split("-")
                if params[0] == "C":
                    # Without Batch Normalization
                    hidden = tf.keras.layers.Conv2D(filters=int(params[1]),
                                                    kernel_size=(int(params[2]), int(params[2])),
                                                    strides=(int(params[3]), int(params[3])), padding=str(params[4]),
                                                    activation='relu')(input_to_layer)
                else:
                    # With Batch Normalization
                    hidden = tf.keras.layers.Conv2D(filters=int(params[1]),
                                                    kernel_size=(int(params[2]), int(params[2])),
                                                    strides=(int(params[3]), int(params[3])), padding=str(params[4]),
                                                    activation=None, use_bias=False)(input_to_layer)
                    hidden = tf.keras.layers.BatchNormalization()(hidden)
                    hidden = tf.keras.layers.Activation('relu')(hidden)

            # MaxPooling Layer
            elif layer.startswith("M"):
                params = layer.split("-")
                hidden = tf.keras.layers.MaxPool2D(pool_size=(int(params[1]), int(params[1])),
                                                   strides=(int(params[2]), int(params[2])))(input_to_layer)

            # Residual Layer
            elif layer.startswith("R"):
                params = layer.split(",")
                params[0] = params[0].split("[")[1]
                params[-1] = params[-1].split("]")[0]
                hidden_start = hidden = input_to_layer
                for cnn in params:
                    cnn_params = cnn.split("-")
                    if params[0] == "C":
                        # Without Batch Normalization
                        hidden = tf.keras.layers.Conv2D(filters=int(cnn_params[1]),
                                                        kernel_size=(int(cnn_params[2]), int(cnn_params[2])),
                                                        strides=(int(cnn_params[3]), int(cnn_params[3])),
                                                        padding=str(cnn_params[4]),
                                                        activation='relu')(hidden)
                    else:
                        # With Batch Normalization
                        hidden = tf.keras.layers.Conv2D(filters=int(cnn_params[1]),
                                                        kernel_size=(int(cnn_params[2]), int(cnn_params[2])),
                                                        strides=(int(cnn_params[3]), int(cnn_params[3])),
                                                        padding=str(cnn_params[4]),
                                                        activation=None, use_bias=False)(hidden)
                        hidden = tf.keras.layers.BatchNormalization()(hidden)
                        hidden = tf.keras.layers.Activation('relu')(hidden)
                hidden = hidden + hidden_start

            # Flatten Layer
            elif layer.startswith("F"):
                hidden = tf.keras.layers.Flatten()(input_to_layer)

            # Dense Hidden Layer
            elif layer.startswith("H"):
                params = layer.split("-")
                hidden = tf.keras.layers.Dense(int(params[1]), activation='relu')(input_to_layer)

            # Dropout Layer
            elif layer.startswith("D"):
                params = layer.split("-")
                hidden = tf.keras.layers.Dropout(float(params[1]))(input_to_layer)

            return hidden

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearty of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in variable `hidden`.
        # Add the final output layer

        # TODO NOTE: I know my technique for joining layers for residual block is inefficient
        #  I realised it late :)

        # Residual Block Handler
        flag = False
        all_layers = args.cnn.split(",")
        for i, j in enumerate(all_layers):
            if j[0] == "R":
                start = i
                for n in range(i + 1, len(all_layers)):
                    if all_layers[n][-1] == "]":
                        end = n
                        flag = True
                        break
        if flag:
            res = []
            for r in range(start, end + 1):
                res.append(all_layers[r])

            for d in range(len(res)):
                del all_layers[start]

            all_layers.insert(start, ",".join(res))

        # ADD LAYERS
        hidden = inputs
        for layer in all_layers:
            hidden = add_layer(layer, hidden)
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100,
                                                          profile_batch=0)
        self.tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    # network.summary()
    # exit()

    network.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[network.tb_callback],
    )

    # Compute test set accuracy and return it
    test_logs = network.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs["accuracy"]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
