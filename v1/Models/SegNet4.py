# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:17:18 2023

@author: Lenovo
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_ORDERING =  "channels_last"

# 编码器网络
def vggnet_encoder(input_height=416, input_width=416, pretrained='imagenet'):

    img_input = tf.keras.Input(shape=(input_height, input_width, 3))

    # 416,416,3 -> 208,208,64
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # 208,208,64 -> 128,128,128
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # 104,104,128 -> 52,52,256
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # 52,52,256 -> 26,26,512
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # 26,26,512 -> 13,13,512
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

# 解码器
def decoder(feature_input, n_classes, n_upSample):
    # feature_input是vggnet第四个卷积块的输出特征矩阵
    # 26,26,512
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(feature_input)
    output = (layers.Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    for _ in range(n_upSample - 2):
        output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
        output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
        output = (layers.Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
        output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,64
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)
	
	# 像素级分类层
    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    output = layers.Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(output)
    print(output)
    
    output = layers.Conv2D(n_classes, (1, 1), activation="sigmoid")(output)
    print(output)

    return output

def SegNet(input_height=256, input_width=256, n_classes=2, n_upSample=3, encoder_level=3):

    img_input, features = vggnet_encoder(input_height=input_height, input_width=input_width)
    feature = features[encoder_level]  # (26,26,512)
    output = decoder(feature, n_classes, n_upSample)

    # 将结果进行reshape
    #output = tf.reshape(output, (-1, int(input_height / 2) * int(input_width / 2), 2))
    #output = layers.sigmoid()(output)
    #Softmax()(output)

    model = tf.keras.Model(img_input, output)

    return model