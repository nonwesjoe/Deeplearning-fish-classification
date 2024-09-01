import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import cv2
#---------------------------------------------
image_folder = '8-fish-classify'
epochs=30
batch_size=32
size = 100
input_shape = (size, size, 3)
num_classes = 8
#---------------
patch_size = 20
num_patches = (size // patch_size) ** 2
projection_dim = 32
num_heads = 3
#---------------------------------------------------------
sub_dirs = os.listdir(image_folder)
images = []
labels = []
for sub_dir in sub_dirs:
    dir_path = os.path.join(image_folder, sub_dir)
    img_files = os.listdir(dir_path)
    for img_file in img_files:
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)
        img= cv2.resize(img, (size, size))
        images.append(img)
        labels.append([int(sub_dir)])

images = np.array(images)/255
labels = np.array(labels)
labels=to_categorical(labels)
#---------------
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=52)

print(y_train.shape, 'ytrain')
print(x_train.shape, 'ytrain')
#-------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
#-------------
def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, num_heads, num_classes):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(3):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=projection_dim, activation="relu")(x3)
        x3 = layers.Dense(units=projection_dim)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = layers.Dense(512, activation="relu")(representation)
    features = layers.Dropout(0.5)(features)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
#-------------------------------------------------------
model = create_vit_classifier(
    input_shape, patch_size, num_patches, projection_dim, num_heads, num_classes
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
checkpoint = ModelCheckpoint(filepath='transformer01.weights.h5',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')
#-----------------------
model.fit(
    x_train,y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test,y_test),
    callbacks=[checkpoint]
)
#---------------------------------------------------------
#0.81


