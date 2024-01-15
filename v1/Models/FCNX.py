# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:43:36 2022

@author: DongXr
"""

from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation, UpSampling2D


def FCN32(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height,input_width,3))
    model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    # 内存原因，卷积核4096时报错OOM，降低至1024
    o = Conv2D(filters=1024,kernel_size=(7,7),padding='same',activation='relu',name='fc6')(model.output)
    o = Dropout(0.5)(o)
    o = Conv2D(filters=1024,kernel_size=(1,1),padding='same',activation='relu',name='fc7')(o)
    o = Dropout(0.5)(o)

    o = Conv2D(filters=nClasses,kernel_size=(1,1),padding='same',activation='relu',name='score_fr')(o)
    o = Conv2DTranspose(filters=nClasses,kernel_size=(32,32),strides=(32,32),padding='valid',activation=None,name='score2')(o)
    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn8 = Model(img_input,o)
    return fcn8

def FCN16(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height,input_width,3))
    # model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    model = FCN32(11, 320, 320)
    model.load_weights("model.h5")


    skip1 = Conv2DTranspose(512,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="upsampling6")(model.get_layer("fc7").output)
    summed = add(inputs=[skip1,model.get_layer("block4_pool").output])
    up7 = UpSampling2D(size=(16,16),interpolation='bilinear',name='upsamping_7')(summed)
    o = Conv2D(nClasses,kernel_size=(3,3),activation='relu',padding='same',name='conv_7')(up7)


    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn16 = Model(model.input,o)
    return fcn16

def FCN8(nClasses,input_height,input_width):

    # model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    model = FCN32(11, 320, 320)
    model.load_weights("model.h5")


    skip1 = Conv2DTranspose(512,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="up7")(model.get_layer("fc7").output)
    # skip2 = Conv2DTranspose(256,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="up4")(model.get_layer('block4_pool').output)

    summed = add(inputs=[skip1,model.get_layer("block4_pool").output])
    skip2 = Conv2DTranspose(256,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name='up4')(summed)

    summed = add(inputs=[skip2,model.get_layer("block3_pool").output])

    up7 = UpSampling2D(size=(8,8),interpolation='bilinear',name='upsamping_7')(summed)
    o = Conv2D(nClasses,kernel_size=(3,3),activation='relu',padding='same',name='conv_7')(up7)


    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn8 = Model(model.input,o)
    return fcn8

