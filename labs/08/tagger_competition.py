#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

# from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--cle_dim", default=128, type=int, help="CLE embedding dimension.")
parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=256, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--word_masking", default=0.2, type=float, help="Mask words with the given probability.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")


class Network(tf.keras.Model):
    def __init__(self, args, train):
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        mapped_words = train.forms.word_mapping(words)
        masked = tf.ones_like(mapped_words, dtype=tf.dtypes.float32)
        masked = tf.keras.layers.Dropout(rate=args.word_masking)(masked)
        masked = tf.cast(masked, dtype=tf.dtypes.int64)
        dropped = tf.math.multiply(mapped_words, masked)
        dropped = tf.cast(dropped, dtype=tf.dtypes.int64)
        embedded_words = tf.keras.layers.Embedding(train.forms.word_mapping.vocab_size(), args.we_dim)(dropped)
        unique_words, indices_words = tf.unique(words.values)
        letters_seq = tf.strings.unicode_split(unique_words, input_encoding="UTF-8")
        mapped_letters_seq = train.forms.char_mapping(letters_seq)
        embedded_chars = tf.keras.layers.Embedding(train.forms.char_mapping.vocab_size(), args.cle_dim)(mapped_letters_seq)
        bidirectional_cle = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=args.cle_dim, return_sequences=False), merge_mode='concat')(embedded_chars.to_tensor(), mask=tf.sequence_mask(embedded_chars.row_lengths()))
        rep_words = tf.gather(bidirectional_cle, indices_words)
        rep_converted = words.with_values(rep_words)
        concat = tf.keras.layers.Concatenate()([embedded_words, rep_converted])

        if args.rnn_cell == "LSTM":
            bidirectional_0 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True), merge_mode='sum')(concat.to_tensor(),mask=tf.sequence_mask(concat.row_lengths()))
            bidirectional = tf.RaggedTensor.from_tensor(bidirectional_0, concat.row_lengths())
        elif args.rnn_cell == "GRU":
            bidirectional_0 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True), merge_mode='sum')(concat.to_tensor(),mask=tf.sequence_mask(concat.row_lengths()))
            bidirectional = tf.RaggedTensor.from_tensor(bidirectional_0, concat.row_lengths())

        predictions = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(train.tags.word_mapping.vocab_size(), activation="softmax"))(bidirectional)

        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")

    # Create the network and train
    model = Network(args, morpho.train)

    # TODO: Construct dataset for training, which should contain pairs of
    # - ragged tensor of string words (forms) as input
    # - ragged tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        return forms, morpho.train.tags.word_mapping(tags)

    train = morpho.train.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dev = morpho.dev.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    test = morpho.test.dataset.map(tagging_dataset).apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))

    model.fit(train, epochs=args.epochs, validation_data=dev)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("tagger_competition.txt", "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
