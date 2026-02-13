import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_autoencoder(input_len=2048):

    inputs = Input(shape=(input_len, 1), name="input_layer")

    branch1_a = layers.Conv1D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)


    branch1_b = layers.Conv1D(16, 9, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)


    concat1 = layers.Concatenate()([branch1_a, branch1_b])

    pool1 = layers.Conv1D(32, 3, strides=2, activation='elu', padding='same', kernel_initializer='he_normal')(concat1)

    branch2_a = layers.Conv1D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool1)


    branch2_b = layers.Conv1D(32, 7, activation='elu', padding='same', kernel_initializer='he_normal')(pool1)


    concat2 = layers.Concatenate()([branch2_a, branch2_b])

    encoded = layers.Conv1D(64, 3, strides=2, activation='elu', padding='same', kernel_initializer='he_normal')(concat2)


    encoded = layers.Dropout(0.1)(encoded)

    x = layers.UpSampling1D(2)(encoded)

    x = layers.Conv1D(64, 7, activation='elu', padding='same', kernel_initializer='he_normal')(x)


    x = layers.UpSampling1D(2)(x)


    x = layers.Conv1D(32, 9, activation='elu', padding='same', kernel_initializer='he_normal')(x)

    outputs = layers.Conv1D(1, 3, activation='sigmoid', padding='same', bias_initializer='zeros', name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CAE_1D")
    return model
