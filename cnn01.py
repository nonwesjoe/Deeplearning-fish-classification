from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input,  Softmax
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import cv2
#---------------------------------------
image_folder = '8-fish-classify'
size=100
num_classes=8
epochs=30
batch_size=32
#-----------------------------------------
sub_dirs = os.listdir(image_folder)
images = []
labels = []
for sub_dir in sub_dirs:
    # 获取文件夹全路径
    dir_path = os.path.join(image_folder, sub_dir)
    # 读取该文件夹下所有图片
    img_files = os.listdir(dir_path)
    for img_file in img_files:
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)
        img= cv2.resize(img, (size, size))
        images.append(img)
        labels.append([int(sub_dir)])

print('done-collecting')
images = np.array(images)/255
labels = np.array(labels)
#-----------
labels=to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=52)

print(y_train.shape, 'ytrain')
print(x_train.shape, 'xtrain')
#-------------------------------------------

model = Sequential([
    # 第一层
    Conv2D(32, (5, 5), padding='valid', activation='relu', input_shape=(size, size,3)),
    Dropout(0.5),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # 第二层
    Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu'),
    Dropout(0.5),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(512,activation='relu'),
    Dense(128,activation='relu'),
    Dense(num_classes, activation='softmax')
])

#--------------
model.summary()

checkpoint =ModelCheckpoint(filepath='cnn01.keras',  # 保存的模型文件名
                                 monitor='val_accuracy',  # 监控的指标，可以选择其他指标如'val_accuracy'
                                 verbose=1,  # 显示保存信息
                                 save_best_only=True,  # 仅保存性能最好的模型
                                 mode='max')
# 指定模型性
model.compile(loss='categorical_crossentropy',                     #loss='binary_crossentropy',   #loss='sparse_categorical_crossentropy'
              optimizer='adam',
              metrics=['accuracy'])
#--------------
model.fit(
    x_train,y_train,batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test,y_test),
    callbacks=[checkpoint]
)
#--------------------------------------------------
