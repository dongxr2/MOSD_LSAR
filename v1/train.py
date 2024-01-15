# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:01:55 2022

@author: Lenovo
"""

from keras.callbacks import ModelCheckpoint#, TensorBoard, EarlyStopping
from multiprocessing import Process
import LoadBatches
#from keras import optimizers
import math
import glob
import tensorflow as tf
from transunet import TransUNet
from keras import backend as K
from keras.losses import binary_crossentropy
from Models import UNet,DeeplabV3Plus,SegNet3,DLinkNet

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 选择ID为0的GPU

gpus = tf.config.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

#############################################################################
def generalized_dice_coefficient(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score
    
def dice_loss(y_true, y_pred):
        loss = 1 - generalized_dice_coefficient(y_true, y_pred)
        return loss

def bce_dice_loss( y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss / 2.0
#############################################################################
#train_path = "tmp/"
key = "funet"

train_batch_size = 4
val_batch_size = 1

n_classes = 2
epochs = 10000

input_height = 256
input_width = 256

loss = bce_dice_loss

i=0

train_images_path="dataset/train/palsar/image/"
train_segs_path="dataset/train/palsar/label/"

val_images_path = "dataset/test/palsar/image/"
val_segs_path = "dataset/test/palsar/label/"

model_path="output/"

tmp= sorted(glob.glob(train_images_path + "*.jpg") + glob.glob(train_images_path + "*.tif") +
            glob.glob(train_images_path + "*.png") + glob.glob(train_images_path + "*.jpeg"))

img_num=len(tmp)

tmp= sorted(glob.glob(val_images_path + "*.jpg") + glob.glob(val_images_path + "*.tif") +
            glob.glob(val_images_path + "*.png") + glob.glob(val_images_path + "*.jpeg"))

val_num=len(tmp)

#此处选择模型 from Models import UNet,DeeplabV3Plus,SegNet3,DLinkNet
#m = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True)
m = DLinkNet.create_dlinknet() #SegNet3.SegNet(256,256,2)#input_height,input_width,n_classes)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
   
m.compile(
    loss='binary_crossentropy',#loss,#'binary_crossentropy', # or loss=loss
    optimizer=opt,
    metrics=[tf.keras.metrics.BinaryAccuracy()]  #binary_accuracy
)

train_set = LoadBatches.imageSegmentationGenerator(train_images_path,
                                           train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)
val_set = LoadBatches.imageSegmentationGenerator(val_images_path,
                                            val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath="output/net.hdf5",
    monitor='val_loss',
    mode='min',
    save_best_only=False,
    save_weights_only=False
    )

m.fit(
    x=train_set,
    steps_per_epoch=img_num // train_batch_size,
    epochs=epochs,
    callbacks=[checkpoint], #, early_stopping
    verbose=1,
    validation_data=val_set,
    validation_steps=val_num//val_batch_size,
    #validation_split=0.2,
    shuffle=True
)
