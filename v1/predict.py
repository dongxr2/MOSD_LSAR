# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:53:11 2022

@author: Lenovo
"""

#import LoadBatches
#from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transunet import TransUNet
import os
from sklearn.cluster import KMeans
import joblib
from Models import UNet,DeeplabV3Plus


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

n_classes = 2
k_num=10
nbr_bins=256
key = "cmtu" #-weights.88"

images_path = "dataset/test/palsar/image/"
segs_path = "dataset/test/palsar/label/"
result_path='dataset/test/palsar/predict/'
output_name="unet"

input_height = 256
input_width = 256

colors = [
    (random.randint(
        0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

##########################################################################

def compute_iou(y_true, y_pred):
     # ytrue, ypred is a flatten vector
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     print(current)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     tmp=np.mean(IoU)
     #print(tmp)
     return tmp


def label2color(colors, n_classes, seg):
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((seg == c) *
                               (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((seg == c) *
                               (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((seg == c) *
                               (colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color


def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy


images = sorted(glob.glob(images_path + "*.jpg") +
                glob.glob(images_path + "*.png") +
                glob.glob(images_path + "*.jpeg") +
                glob.glob(images_path + "*.tif"))

segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                       glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg") +
                       glob.glob(segs_path + "*.tif")
                       )

output_path="./output/"

output_models= sorted(glob.glob(output_path + output_name+".hdf5"))
print(output_models)

'''
#单个模型输出
for mi in output_models:
    if output_name in mi:
        m = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True)
        m.load_weights(mi)
        seg_list,pr_list=np.array([]).astype('uint8'),np.array([]).astype('uint8')
        for i, (imgName, segName) in enumerate(zip(images, segmentations)):
            im = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
            
            seg = cv2.imread(segName, 0)
            
            pr = m.predict(np.expand_dims(im, 0))[0]
            pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
            
            seg_list=(seg.flatten()/255).astype('uint8')
            pr_list=pr.flatten().astype('uint8')
            
            pr = pr * 255.0
            cv2.imwrite(result_path+os.path.basename(imgName).split('.')[0]+'_im.png', im)
            cv2.imwrite(result_path+os.path.basename(imgName).split('.')[0]+'_pre.png', pr)
            cv2.imwrite(result_path+os.path.basename(imgName).split('.')[0]+'_gt.png', seg)
            try:
                f = open ("result_"+output_name+".txt", "a")
                print(imgName+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
                      str(precision_score(seg_list,pr_list))+"\t"+
                      str(recall_score(seg_list,pr_list))+"\t"+
                      str(f1_score(seg_list,pr_list))+"\t"+
                      str(compute_iou(seg_list,pr_list)),file=f
                      )    
                f.close()
            except Exception as r:
                f = open ("result_"+output_name+".txt", "a")
                print(mi,r,file=f)
                f.close()
'''
#全部计算
for mi in output_models:
    try:
        m = DeeplabV3Plus.DeeplabV3Plus(n_classes, input_height=input_height, input_width=input_width)
        #m = UNet.unet_fengfan(n_classes, input_height=input_height, input_width=input_width)
        #m = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True)
        m.load_weights(mi)
        seg_list,pr_list=np.array([]).astype('uint8'),np.array([]).astype('uint8')
        for i, (imgName, segName) in enumerate(zip(images, segmentations)):
            im = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
            
            seg = cv2.imread(segName, 0)
            
            pr = m.predict(np.expand_dims(im, 0))[0]
            pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
            
            seg_list=np.hstack((seg_list,(seg.flatten()/255).astype('uint8')))
            pr_list=np.hstack((pr_list,pr.flatten().astype('uint8')))

        f = open ("result_"+output_name+".txt", "a")
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list))+"\t"+
              str(recall_score(seg_list,pr_list))+"\t"+
              str(f1_score(seg_list,pr_list))+"\t"+
              str(compute_iou(seg_list,pr_list)),file=f
              )    
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(recall_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(f1_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(compute_iou(seg_list,pr_list))
              ) 
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(recall_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(f1_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(compute_iou(seg_list,pr_list))
              )  
        f.close()
    except Exception as r:
        f = open ("result_"+output_name+".txt", "a")
        print(mi,r,file=f)
        f.close()
