import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class VAELossLayer(layers.Layer):
    def call(self, inputs):
        true_inputs, reconstruction, z_mean, z_log_var = inputs
        rec_loss = tf.reduce_mean(tf.square(true_inputs - reconstruction))
        rec_loss *= 2048
        # 2. KL Divergence
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        self.add_loss(rec_loss + kl_loss)
        return reconstruction
def build_vae(input_len=2048, latent_dim=16):
    encoder_inputs = tf.keras.Input(shape=(input_len, 1), name="input_vae")
    x = layers.Conv1D(32, 3, activation="relu", padding="same", strides=2)(encoder_inputs)
    x = layers.Conv1D(64, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(256 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((256, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    z_mean, z_log_var, z = encoder(encoder_inputs)
    reconstruction = decoder(z)
    outputs = VAELossLayer()([encoder_inputs, reconstruction, z_mean, z_log_var])
    vae = models.Model(encoder_inputs, outputs, name="VAE_Reference_Hones")
    return vae
