
import os, tensorflow as tf
from tensorflow.keras import layers

path = "../data/"

for split in ["training", "validation"]:
    print(f"\n{split.upper()} klasörü:")
    split_path = os.path.join(path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        num_files = len(os.listdir(class_path))
        print(f"  {class_name}: {num_files} dosya")

batch_size = 32
target_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(path, "training"),
    image_size=target_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(path, "validation"),
    image_size=target_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)


class Sampling(layers.Layer):
    def __init__(self, beta_init=0.0, tau=0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta = tf.Variable(beta_init, trainable=False, dtype=tf.float32)
        self.tau  = tf.constant(tau, dtype=tf.float32)

    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        z   = z_mean + tf.exp(0.5 * z_logvar) * eps

        # KL per-dim
        kl_per_dim = -0.5 * (1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        # free-bits (base threshold)
        kl_per_dim = tf.maximum(kl_per_dim, self.tau)

        # batch mean + latent_dim normalize
        kl = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=-1))
        latent_dim = tf.cast(tf.shape(z_mean)[-1], tf.float32)
        kl = kl / latent_dim

        self.add_loss(self.beta * kl)
        return z

class BetaAnneal(tf.keras.callbacks.Callback):
    def __init__(self, sampler_layer, final_beta=0.01, warmup_epochs=20):
        super().__init__()
        self.sampler = sampler_layer
        self.final_beta = float(final_beta)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        t = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.sampler.beta.assign(self.final_beta * t)


def train_model(latent_dim, beta_final):
    # CNN Backbones
    ConvNeXtLarge = tf.keras.applications.ConvNeXtLarge(input_shape=(224,224,3), include_top=False, weights="imagenet")
    VGG19        = tf.keras.applications.VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")
    VGG16        = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights="imagenet")
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    ResNet50     = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights="imagenet")
    for m in [ConvNeXtLarge,VGG19,VGG16,mobilenet,ResNet50]:
        m.trainable = False

    inp  = tf.keras.Input(shape=(224,224,3))
    outs = [m(inp, training=False) for m in [ConvNeXtLarge,VGG19,VGG16,mobilenet,ResNet50]]

    # Downsampling with GAP
    outs = [layers.GlobalAveragePooling2D()(o) for o in outs]
    x = layers.Concatenate()(outs)

    # projection + head
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512,  activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # VAE bottleneck
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_log_var")(x)

    sampler = Sampling(beta_init=0.0, tau=0.5, name="z")  # beta will be initiated from 0
    z = sampler([z_mean, z_logvar])

    h = layers.Dense(512, activation="relu")(z)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(25, activation="softmax")(h)

    model = tf.keras.Model(inp, out, name="convnext_vae_classifier")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    opt     = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    cbs = [
        BetaAnneal(sampler, final_beta=beta_final, warmup_epochs=20),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        callbacks=cbs,
        verbose=1
    )
    return model, history



if __name__ == "__main__":
    '''
    example usages
    train_model(32,0.05)
    train_model(32,0.01)
    train_model(32,0.001)
    train_model(128,0.05)
    train_model(128,0.01)
    train_model(128,0.001)
    train_model(256,0.05)
    train_model(256,0.01)
    train_model(256,0.001)
    '''