#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=24, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Network(tf.keras.Model):
    def __init__(self, args, train):
        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        mapped_words = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.
        embedded_words = tf.keras.layers.Embedding(train.forms.word_mapping.vocab_size(), args.we_dim)(mapped_words)

        # TODO(tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the embedded words, **summing** the outputs of forward and backward RNNs.
        if args.rnn_cell == "LSTM":
            bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True), merge_mode='sum')(embedded_words.to_tensor(), mask=tf.sequence_mask(embedded_words.row_lengths()))
            bidirectional = tf.RaggedTensor.from_tensor(bidirectional, embedded_words.row_lengths())
        elif args.rnn_cell == "GRU":
            bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True), merge_mode='sum')(embedded_words.to_tensor(), mask=tf.sequence_mask(embedded_words.row_lengths()))
            bidirectional = tf.RaggedTensor.from_tensor(bidirectional, embedded_words.row_lengths())

        # TODO: Add a final classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that **no activation** should
        # be used, the CRF operations will take care of it. Also do not forget to use
        # `tf.keras.layers.TimeDistributed`.
        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(train.tags.word_mapping.vocab_size()))(bidirectional)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)

        # We compile the model without loss, because `train_step` will directly call
        # the `selt.crf_loss` method.
        self.compile(optimizer=tf.optimizers.Adam(),
                     metrics=[self.SpanLabelingF1Metric(train.tags.word_mapping.get_vocabulary(), name="f1")])

        # TODO: Create `self._crf_weights`, a trainable zero-initialized tf.float32 matrix variable
        # of size [number of unique train tags, number of unique train.tags], using `self.add_weight`.
        self._crf_weights = self.add_weight(
            'crf_weights',
            (train.tags.word_mapping.vocab_size(), train.tags.word_mapping.vocab_size()),
            dtype=tf.float32,
            trainable=True,
            initializer='zeros'
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.


    def crf_loss(self, gold_labels, logits):
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CRF loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CRF loss must be RaggedTensors"

        # TODO: Use `tfa.text.crf_log_likelihood` to compute the CRF log likelihood.
        # You will have to convert both logits and gold_labels to dense Tensors and
        # use `gold_labels.row_lengths()` as `sequence_length`. Use the `self._crf_weights`
        # as the transition weights.
        #
        # Finally, compute the loss using the computed log likelihoods, averaging the
        # individual batch examples.
        sequence_length = gold_labels.row_lengths()
        logits_dense = logits.to_tensor()
        gold_labels_dense = gold_labels.to_tensor()

        loss, _ = tfa.text.crf_log_likelihood(logits_dense, gold_labels_dense, sequence_length, self._crf_weights)
        return tf.reduce_mean(loss)

    def crf_decode(self, logits):
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO: Perform CRF decoding using `tfa.text.crf_decode`. Convert the
        # logits analogously as in `crf_loss`. Finally, convert the result
        # to a ragged tensor.
        #
        # Note: ignore the warning generated by tensorflow_addons/text/crf.py:540.
        # It does not apply to us, because we are passing a regular tensor to it.
        sequence_length = logits.row_lengths()
        logits = logits.to_tensor()
        (sparse_predictions,), _ = tfa.text.crf_decode(logits.to_tensor(), self._crf_weights, sequence_length)
        predictions = tf.RaggedTensor.from_tensor(sparse_predictions)

        assert isinstance(predictions, tf.RaggedTensor)
        return predictions

    # We override the `train_step` method, because:
    # - computing losses on RaggedTensors is not supported in TF 2.4
    # - we do not want to evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.crf_loss(y, y_pred)
            if self.losses: # Add regularization losses if present
                loss += tf.math.add_n(self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # We override `predict_step` to run CRF decoding during prediction
    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.crf_decode(y_pred)
        return y_pred

    # We override `test_step` to use `predict_step` to obtain CRF predictions.
    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    class SpanLabelingF1Metric(tf.metrics.Metric):
        """Keras-like metric evaluating span labeling F1-score of RaggedTensors."""
        def __init__(self, tags, name="span_labeling_f1", dtype=None):
            super().__init__(name, dtype)
            self._tags = tags
            self._counts = self.add_weight("counts", shape=[3], initializer=tf.initializers.Zeros(), dtype=tf.int64)

        def reset_states(self):
            self._counts.assign([0] * 3)

        def classify_spans(self, y_true, y_pred, sentence_limits):
            sentence_limits = set(sentence_limits)
            spans_true, spans_pred = set(), set()
            for spans, labels in [(spans_true, y_true), (spans_pred, y_pred)]:
                span = None
                for i, label in enumerate(self._tags[label] for label in labels):
                    if span and (label.startswith(("O", "B")) or i in sentence_limits):
                        spans.add((start, i, span))
                        span = None
                    if label.startswith("B"):
                        span, start = label[2:], i
                if span:
                    spans.add((start, len(labels), span))
            return np.array([len(spans_true & spans_pred), len(spans_pred - spans_true), len(spans_true - spans_pred)], np.int64)

        def update_state(self, y_true, y_pred, sample_weight=None):
            assert isinstance(y_true, tf.RaggedTensor) and isinstance(y_pred, tf.RaggedTensor)
            assert sample_weight is None, "sample_weight currently not supported"
            counts = tf.numpy_function(self.classify_spans, (y_true.values, y_pred.values, y_true.row_limits()), tf.int64)
            self._counts.assign_add(counts)

        def result(self):
            tp, fp, fn = self._counts[0], self._counts[1], self._counts[2]
            return tf.math.divide_no_nan(tf.cast(2 * tp, tf.float32), tf.cast(2 * tp + fp + fn, tf.float32))


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
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args, morpho.train)

    # TODO(tagger_we): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        return forms, morpho.train.tags.word_mapping(tags)

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    # Return train loss and dev set accuracy for ReCodEx to validate
    return logs.history["loss"][-1], logs.history["val_f1"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
