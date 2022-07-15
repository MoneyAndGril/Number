import time
from PIL import Image
import keras_preprocessing.image
from keras import callbacks
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import cv2
import numpy as np
from keras import models
from keras import layers
#  修改  from keras.utils import to_categorical  # 转换成 One_hot编码
from tensorflow.keras.utils import to_categorical

from tensorflow.python.keras.optimizer_v1 import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(type(train_images[0]))
# print(train_images[0])
print(train_labels.shape)
zero_path = './pic\\0'
zero_list = []
file_zero = os.listdir(zero_path)
count = 0
for zero in file_zero:
    count += 1
    addr = os.path.join(zero_path,zero)
    # print(addr)
    img_zero = keras_preprocessing.image.load_img(addr,target_size=(28,28))
    img_zero = img_zero.convert('L')
    img_numpy = keras_preprocessing.image.img_to_array(img_zero)
    img_numpy = np.reshape(img_numpy,[28,28,1])
    zero_list.append(img_numpy)
print(count)
zero_train = np.array(zero_list)
zero_train = zero_train / 255.0
print(zero_train.shape)
zero_label = np.array([0])
zero_label = np.repeat(zero_label,10,axis=0)
print(zero_label.shape)
nine_path = './pic\\9'
file = os.listdir(nine_path)
count = 0
list_nine = []
for img_nine in file:
    count = count + 1
    addr = os.path.join(nine_path,img_nine)
    # print(addr)
    img = keras_preprocessing.image.load_img(addr,target_size=(28,28))
    img = img.convert('L')
    img_numpy = keras_preprocessing.image.img_to_array(img)
    # print(img_numpy.shape)
    # img_numpy.show()
    # cv2.imshow('img',img_numpy)
    # cv2.waitKey(0)
    # im = Image.fromarray(img_numpy,'L')
    # im = Image.fromarray(np.uint8(img_numpy)).convert('L')
    # im.show()
    # time.sleep(25)
    # break
    nine_train = np.reshape(img_numpy, [28, 28, 1])
    list_nine.append(img_numpy)

    # print(nine_train.shape)
    # break
nine_train = np.array(list_nine) / 255.0
one_label = np.array([9])
nine_label = np.repeat(one_label,270,axis=0)
# print(line_label)
# exit(0)
print(nine_train.shape)
# print(count)
# exit(0)
# img = np.array(train_images[0])
# img_gray = np.resize(np.array,[28,28,1])
# img = np.repeat(img_gray,3,axis=-1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# name = 1
# for img in test_images:
#     img = np.reshape(img,[28,28,1])
#     img = np.repeat(img, 3, axis=-1)
#     cv2.imwrite(f'D:\\python_game\\game_hyol\\MyMinst\\pic\\mnist_pic\\{name}.jpg',img)
#     if name == 50:
#         break
#     name += 1

# print(train_labels[0])   # 不是One_hot
# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

train_x = train_images.reshape(-1,28,28,1)
test_x = test_images.reshape(-1,28,28,1)

train_x = np.concatenate((train_x,nine_train,zero_train),axis=0)

print(train_x.shape)
# exit(0)
# np.repeat(train_x,)
# train_x = np.concatenate((train_x,train_x,train_x),axis=-1)
# train_x = np.concatenate((test_x,test_x,test_x),axis=-1)

# print(train_x.shape)
# exit(0)

# print(train_images.shape)
train_labels = np.concatenate((train_labels,nine_label,zero_label),axis=-1)
# print(train_labels.shape)
# exit(0)
train_y = to_categorical(train_labels,num_classes=10)
test_y = to_categorical(test_labels,num_classes=10)
# print(train_y[0])    # 5
# print(test_y[0])     # 7


# class Net():

# resNet = keras.applications.resnet.ResNet50(include_top=False,     # 改成自己的全连接
#                                    weights='imagenet',
#                                    input_shape=(28,28,3),
#                                    pooling='avg'
#                                    )
                                   # classes=1000)

# resNet.summary()
model = models.Sequential()
# model.add(resNet())
# kernel_regularizer=keras.regularizers.l2(0.001)
# bias_regularizer=keras.regularizers.l1(0.001)
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',strides=1,input_shape=(28,28,1)))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same',strides=1))
model.add(layers.MaxPool2D((2,2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(rate=0.5))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same',strides=1))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same',strides=1))
model.add(layers.MaxPool2D((2,2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(rate=0.5))
# model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001),bias_regularizer=keras.regularizers.l2(0.001)))
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

# adam rmsprop  sgd
# optimizer='rmsprop'
# # optimizer=Adam(lr=1e-4)
save_weight = './weights.{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkPoint = callbacks.ModelCheckpoint(filepath=save_weight,monitor='val_accuracy',save_weights_only=False,save_best_only=False)
# early_stopping = callbacks.EarlyStopping(patience=15,restore_best_weights=True)
reduce_lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                factor=0.5,
                                                patience=3,
                                                verbose=1,
                                                )
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# shuffle=True
model.fit(train_x,train_y,epochs=200,batch_size=128,validation_data=(test_x,test_y),callbacks=[reduce_lr_plateau,checkPoint])
test_loss,test_acc = model.evaluate(test_x,test_y)

path='./net.hdf5'
model.save(path)
print('保存了')








