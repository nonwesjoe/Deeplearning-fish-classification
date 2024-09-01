from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------
#modelpath='./models/cnn01.keras'
#modelpath='./models/cnn02.keras'
modelpath='./models/cnn03.keras'
model=load_model(modelpath)
size = 100
input_shape = (size, size, 3)
num_classes = 8
path='8-fish-classify-test'
#-------------------------------------------------------

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
    picc = load_img(picpath, target_size=(size, size))
    plt.imshow(picc)
    picc = img_to_array(picc) / 255.0  # 转换为 NumPy 数组并归一化
    picc= np.expand_dims(picc, axis=0)  # 添加批次维度
    predictions = model.predict(picc)
    predicted_class = np.argmax(predictions[0])
    fishname = [key for key, values in fishdic.items() if predicted_class == values]
    plt.title(f"{fishname}")
    plt.show()
    print(f"Predicted class: {predicted_class} -{fishname} for the image  {picpath}")
#---------------------------------------------------------

