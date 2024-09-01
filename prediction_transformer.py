from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
#modelpath='./models/transformer01.weights.h5'
modelpath='./models/transformer02.weights.h5'
#modelpath='./models/transformer03.weights.h5'
#---------------------------------------------
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
model.load_weights(modelpath)

path='8-fish-classify-test'

fishdic={'Chanos Chanos' :1,
'Eleutheronema Tetradactylum': 2,
'Johnius Trachycephalus' :3,
'Nibea Albiflora': 4,
'Oreochromis Mossambicus': 5,
'Oreochromis Niloticus': 6,
'Rastrelliger Faughni' :7,
'Upeneus Moluccensis': 0}

use=[]
for pic in os.listdir(path):
    picpath=os.path.join(path,pic)
    picc = load_img(picpath, target_size=(100, 100))
    plt.imshow(picc)
    picc = img_to_array(picc) / 255.0  # 转换为 NumPy 数组并归一化
    picc= np.expand_dims(picc, axis=0)  # 添加批次维度
    predictions = model.predict(picc)
    predicted_class = np.argmax(predictions[0])
    fishname = [key for key, values in fishdic.items() if predicted_class == values]
    plt.title(fishname)
    plt.show()
    print(f"Predicted class: {predicted_class} {fishname}for the pic  {picpath}")
#---------------------------------------------------------



