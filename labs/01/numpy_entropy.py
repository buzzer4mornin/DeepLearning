#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    with open("numpy_entropy_data.txt", "r") as data:
        ddict = {}
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            try:
                ddict[line] += 1
            except KeyError:
                ddict[line] = 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    arr = np.array(list(ddict.values()))
    data_dist = arr / sum(arr)

    # TODO: Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        model_dist = []
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            model_dist.append(float(line[-3:]))

    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array(model_dist)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = round(-np.sum(data_dist * np.log(data_dist)), 2)

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    data_dist = np.concatenate((data_dist, np.zeros(1)))
    crossentropy = round(-np.sum(data_dist * np.log(model_dist)), 2)

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = round((crossentropy - entropy), 2)

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
