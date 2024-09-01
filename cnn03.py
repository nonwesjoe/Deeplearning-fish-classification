from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#---------------------------------------
image_folder = '8-fish-classify'
size=100
num_classes=8
epochs=30
batch_size=32
#-----------------------------------------
# 使用 ImageDataGenerator 进行数据增强和预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    image_folder,
    target_size=(size, size),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    image_folder,
    target_size=(size, size),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)
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

#-------------
model.summary()

checkpoint =ModelCheckpoint(filepath='cnn03.keras',  # 保存的模型文件名
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
    train_generator,batch_size=batch_size,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[checkpoint]
)
#--------------------------------------------------
