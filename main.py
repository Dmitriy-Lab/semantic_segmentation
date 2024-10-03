pip install opendatasets

import tensorflow as tf                                                               
import keras                                                                          
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Rescaling     
from keras.layers import BatchNormalization, Conv2DTranspose, Concatenate             
from keras.layers import Rescaling, Resizing                                          
from keras.models import Model, Sequential                                            

from keras.optimizers import Adam                                                     
from keras.preprocessing.image import  load_img                                       
from keras.utils import to_categorical                                                

import random                                                                         

import numpy as np                                                                    
import pandas as pd                                                                   
import os                                                                             
import albumentations as A                                                            

import matplotlib.pyplot as plt                                                       
import opendatasets as op

op.download("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/")
dataset_path = './covid19-radiography-database/COVID-19_Radiography_Dataset/Normal'

image_dir = 'images'
label_dir = 'masks'

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Входное изображение', 'Оригинальная маска', 'Предсказанная маска']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[0]))            
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]),alpha=0.5)  
    plt.axis('off')
  plt.show()                                                            

original_image = os.path.join(dataset_path, image_dir, 'Normal-2.png')       
label_image_semantic = os.path.join(dataset_path, label_dir, 'Normal-2.png') 

fig, axs = plt.subplots(1, 2, figsize=(16, 8))                          

img = np.array(load_img(original_image, target_size=(256, 256), color_mode='rgb'))   
mask = np.array(load_img(label_image_semantic, target_size=(256, 256), color_mode='grayscale'))  

axs[0].imshow(img)  
axs[0].grid(False)

axs[1].imshow(mask) 
axs[1].grid(False)

input_img_path = sorted(
    [
        os.path.join(dataset_path, image_dir, fname)
        for fname in os.listdir(os.path.join(dataset_path, image_dir))
        if fname.endswith(".png")
    ]
)

target_img_path = sorted(
    [
        os.path.join(dataset_path, label_dir, fname)
        for fname in os.listdir(os.path.join(dataset_path, label_dir))
        if fname.endswith(".png")
    ]
)

batch_size = 16
img_size = (256, 256)
num_classes = 2 # 2 класса: фон и сегментированый объект

class datasetGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_path, target_img_path, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_path = input_img_path
        self.target_img_path = target_img_path


    def __len__(self):
        """Возвращает число мини-батчей обучающей выборки"""
        return len(self.target_img_path) // self.batch_size


    def __getitem__(self, idx):
        """Возвращает кортеж (input, target) соответствующий индексу пакета idx"""

        # Формируем пакеты из ссылок путем среза длинной в batch_size и возвращаем пакет по индексу
        batch_input_img_path = self.input_img_path[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_target_img_path = self.target_img_path[idx*self.batch_size:(idx+1)*self.batch_size]

        # Создадим массив numpy, заполненный нулями, для входных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        x = np.zeros((self.batch_size, *self.img_size, 3), dtype="float32")

        # Создадим массив numpy, заполненный нулями, для выходных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        y = np.zeros((self.batch_size, *self.img_size, num_classes), dtype="uint8")

        # В цикле заполняем массивы с изображениями x и y
        # Перебираем пакеты из путей batch_input_img_path и batch_target_img_path к изображениям
        # zip возвращает для нескольких последовательностей список кортежей из элементов последовательностей с одинаковыми индексами
        for _, paths in enumerate(zip(batch_input_img_path, batch_target_img_path)):

            img = np.array(load_img(paths[0], target_size=self.img_size, color_mode='rgb'))         
            mask = np.array(load_img(paths[1], target_size=self.img_size, color_mode='grayscale'))  
            mask = mask / 255
            x[_] = img / 255 
            y[_] = to_categorical(mask, num_classes=num_classes) 

        return x, y
      

train_input_img_path = input_img_path[:1500]
train_target_img_path = target_img_path[:1500]

val_input_img_path = input_img_path[1500:1700]
val_target_img_path = target_img_path[1500:1700]

train_gen = datasetGenerator(batch_size, img_size, train_input_img_path, train_target_img_path, num_classes)

val_gen = datasetGenerator(batch_size, img_size, val_input_img_path, val_target_img_path, num_classes)

def convolution_operation(entered_input, filters=64):

    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    acti1 = ReLU()(batch_norm1)


    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(acti1)
    batch_norm2 = BatchNormalization()(conv2)
    acti2 = ReLU()(batch_norm2)

    return acti2

def encoder(entered_input, filters=64):

    encod1 = convolution_operation(entered_input, filters) 
    MaxPool1 = MaxPooling2D(strides = (2,2))(encod1)        
    return encod1, MaxPool1 

def decoder(entered_input, skip, filters=64):
    Upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input) 
    Connect_Skip = Concatenate()([Upsample, skip])                                        
    out = convolution_operation(Connect_Skip, filters)                                    
    return out 
  
# модель U-net
def U_Net(img_size, num_classes):
    # Входной слой - желтый блок
    inputs = Input(img_size)

    
    skip1, encoder_1 = encoder(inputs, 64)
    skip2, encoder_2 = encoder(encoder_1, 64*2)
    skip3, encoder_3 = encoder(encoder_2, 64*4)
    skip4, encoder_4 = encoder(encoder_3, 64*8)


    conv_block = convolution_operation(encoder_4, 64*16)

    
    decoder_1 = decoder(conv_block, skip4, 64*8)
    decoder_2 = decoder(decoder_1, skip3, 64*4)
    decoder_3 = decoder(decoder_2, skip2, 64*2)
    decoder_4 = decoder(decoder_3, skip1, 64)

   
    outputs = Conv2D(num_classes, kernel_size = (1, 1), padding="same", activation="softmax")(decoder_4)

    model = Model(inputs, outputs)
    return model

input_shape = (img_size[0], img_size[1], 3)
model = U_Net(input_shape, num_classes) 

model.compile(
    optimizer='adam' ,
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint("segmentation.keras", monitor='val_loss', save_best_only=True)
]

epochs = 5
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                   )

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Входное изображение', 'Оригинальная маска', 'Предсказанная маска']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[0])            # отображаем картинку
    plt.imshow(display_list[i],alpha=0.8)  # отображаем маску с прозрачностью 50%
    plt.axis('off')
  plt.show()


for index in range(10):               
    img = np.array(load_img(val_input_img_path[index], target_size=(256, 256), color_mode='rgb')) 
    mask = np.array(load_img(val_target_img_path[index], target_size=(256, 256), color_mode='grayscale'))

    test = model.predict(np.expand_dims(img, 0) / 255)

    test = np.argmax(test, axis=-1)

    display([img.reshape(1, 256, 256, 3)[0], mask, test[0]])  

