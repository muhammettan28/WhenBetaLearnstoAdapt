
import os
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import layers

print(tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)


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
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(path, "validation"),
    image_size=target_size,
    batch_size=batch_size,
    label_mode='int'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

class Sampling(layers.Layer):
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    def call(self, inputs):
        z_mean, z_logvar = inputs
        # reparametrization: z = μ + σ ⊙ ε
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_logvar) * epsilon
        kl = -0.5 * tf.reduce_sum(1. + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1)
        self.add_loss(self.beta * tf.reduce_mean(kl))
        return z

def train_model(latent_dim, beta):
    ConvNeXtLarge = tf.keras.applications.ConvNeXtLarge(input_shape=(224,224,3), include_top=False, weights="imagenet")
    VGG19        = tf.keras.applications.VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")
    VGG16        = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights="imagenet")
    Xception     = tf.keras.applications.Xception(input_shape=(224,224,3), include_top=False, weights="imagenet")
    ResNet50     = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights="imagenet")

    for m in [ConvNeXtLarge,VGG19,VGG16,Xception]:
        m.trainable = False
        m.VGG19=False
        m.VGG16=False
        m.Xception=False

    inp = tf.keras.Input(shape=(224,224,3))
    outs = [m(inp, training=False) for m in [ConvNeXtLarge,VGG19,VGG16,Xception]]

    # More stable ve lightweight: GAP instead of Flatten
    outs = [layers.GlobalAveragePooling2D()(o) for o in outs]
    x = layers.Concatenate()(outs)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512,  activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(beta=beta, name="z")([z_mean, z_logvar])

    h = layers.Dense(1024, activation="relu")(z)
    out = layers.Dense(25, activation="softmax")(h)

    model = tf.keras.Model(inp, out, name="convnext_vae_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]
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