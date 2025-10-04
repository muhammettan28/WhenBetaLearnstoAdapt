import os
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

path = "../data/"

for split in ["training", "validation"]:
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

# You can use other Pre-trained CNN Models as well
ConvNeXtLarge = tf.keras.applications.ConvNeXtLarge(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
ConvNeXtLarge.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = ConvNeXtLarge(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(11, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=150,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,monitor="val_loss")]
)

