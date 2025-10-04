
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers

print(tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

path = "../data/"

for split in ["training", "validation"]:
    print(f"\n{split.upper()} directory:")
    split_path = os.path.join(path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        num_files = len(os.listdir(class_path))
        print(f"  {class_name}: {num_files} files")


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


ConvNeXtLarge = tf.keras.applications.ConvNeXtLarge(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
ConvNeXtLarge.trainable = False

vgg19 = tf.keras.applications.VGG19(
    input_shape=(224, 224, 3), 
    include_top=False, weights='imagenet')
vgg19.trainable = False

vgg16 = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
      include_top=False,
        weights='imagenet')
vgg16.trainable = False

Xception = tf.keras.applications.Xception(
    input_shape=(224, 224, 3),
      include_top=False,
        weights='imagenet')
Xception.trainable = False

Resnet = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet')
Resnet.trainable = False

latent_dim = 256 # you can choose 32, 128 as well

input_layer = tf.keras.Input(shape=(224, 224, 3))
vgg19_output = vgg19(input_layer, training=False)
ConvNeXtLarge_output = ConvNeXtLarge(input_layer, training=False)
vgg16_output = vgg16(input_layer, training=False)
xception_output = Xception(input_layer, training=False)
resnet_output = Resnet(input_layer, training=False)
flattened_outputs = [layers.Flatten()(model_output) for model_output in [vgg19_output, ConvNeXtLarge_output,vgg16_output,xception_output,resnet_output]]

merged_output = layers.concatenate(flattened_outputs)


x = tf.keras.layers.Dense(512, activation="relu")(merged_output)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)


latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent_bottleneck")(x)
x = tf.keras.layers.Dense(256, activation="relu")(latent)
outputs = tf.keras.layers.Dense(25, activation="softmax")(x)
model = tf.keras.Model(input_layer, outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=150,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,monitor="val_loss")]
)