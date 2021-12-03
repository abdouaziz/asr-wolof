import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from data import load_data
from model import ASR
from loss import compute_ctc_loss


def parse_args():

    parser = argparse.ArgumentParser(
        description="Pretrained Machine Translation French to Wolof")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv  file containing the training data."
    )

    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv file containing the development data."
    )

    parser.add_argument(
        "--audio_dir", type=str, default=None, help="The path to the audio files."
    )

    parser.add_argument(
        "--n_filters", type=int, default=256, help="Number of filters in the convolutional layers."
    )

    parser.add_argument(
        "--kernel_size", type=int, default=11, help="Size of the kernel in the convolutional layers."
    )

    parser.add_argument(
        "--conv_stride", type=int, default=2, help="Stride of the convolutional layers."
    )

    parser.add_argument(
        "--conv_border", type=str, default="valid", help="Border mode of the convolutional layers."
    )

    parser.add_argument(
        "--n_lstm_units", type=int, default=256, help="Number of units in the LSTM layers."
    )

    parser.add_argument(
        "n_dense_units", type=int, default=42, help="Number of units in the dense layers."
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="The path to the output directory."
    )

    args = parser.parse_args()

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in [
            "csv", "json"], "`train_file` should be a csv or a json file."

    if args.dev_file is not None:
        extension = args.dev_file.split(".")[-1]
        assert extension in [
            "csv", "json"], "`dev_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()

    dataset = load_data(args)

    # Create the model

    model = ASR(256, 11, 2, 'valid', 256, 42)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(args.epochs):

        for step, d in enumerate(dataset):
            with tf.GradientTape() as tape:

                logits = model(d[0])
                labels = d[1]
                logits_length = [logits.shape[1]*logits.shape[0]]
                labels_length = [labels.shape[1]*labels.shape[0]]
                loss = compute_ctc_loss(
                    logits, labels, logit_length=logits_length, label_length=labels_length)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("Epoch: {}, Step: {}, Loss: {}".format(
                    epoch, step, loss.numpy()))

    model.save_weights(args.output_dir + '/model.h5')

    # Evaluate the model

    # model.evaluate(dataset)
    if args.dev_file is not None:
        pass


if __name__ == "__main__":
    main()
