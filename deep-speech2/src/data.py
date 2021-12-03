import tensorflow as tf
import pandas as pd
from engine import generate_input_from_audio_file, generate_target_output_from_text


def load_data(args):
    """
    Load data from file and generate the dataset.
    """
    data = pd.read_csv(args.train_file)

    def generator():

        for i in range(len(data)):

            audio_path = data.iloc[i]["ID"]
            transcript = data.iloc[i]["transcription"]

            audio = generate_input_from_audio_file(
                args.audio_dir+"/"+audio_path+".mp3")
            audio = tf.expand_dims(audio, axis=0)

            label = generate_target_output_from_text(transcript, args)
            label = tf.expand_dims(tf.convert_to_tensor(label), axis=0)

        yield audio, label

    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.int32),
                                             (tf.TensorShape([None, None, None]), tf.TensorShape([1, None])))
    #dataset = dataset.repeat(c)

    #dataset = dataset.padded_batch(config.batch_size, (tf.TensorShape([None, config.input_dim, 1]), tf.TensorShape([None])))

    dataset = dataset.batch(args.batch_size).prefetch(10)

    return dataset
