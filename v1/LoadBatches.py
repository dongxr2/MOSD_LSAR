import numpy as np
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import random

'''
def getImageArr(im):

    img = im.astype(np.float32)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img
'''

def getSegmentationArr(seg, nClasses, input_height, input_width):

    seg_labels = np.zeros((input_height, input_width, nClasses))

    for c in range(nClasses):
        seg_labels[:, :, c] = (seg == c).astype(int)

    # seg_labels = np.reshape(seg_labels, (-1, nClasses))
    return seg_labels


def imageSegmentationGenerator_backup(images_path, segs_path, batch_size,
                               n_classes, input_height, input_width):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.tif") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.tif") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            # print(im, seg)
            im = cv2.imread(im, cv2.IMREAD_UNCHANGED)
            # print(im.max(), im.min())
            # cv2.imwrite('img.tif', im)
            # break
            seg = cv2.imread(seg, 0)
            # print(seg.shape)

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] >= input_height and im.shape[1] >= input_width

            xx = random.randint(0, im.shape[0] - input_height)
            yy = random.randint(0, im.shape[1] - input_width)

            im = im[xx:xx + input_height, yy:yy + input_width]
            seg = seg[xx:xx + input_height, yy:yy + input_width]

            # print(seg.max(), seg.min())
            X.append(im)
            if seg.max() > 254:
                seg = seg / 255
            # print(seg.max(), seg.min())
            Y.append(
                getSegmentationArr(
                    seg,
                    n_classes,
                    input_height,
                    input_width))

        yield np.array(X), np.array(Y)

def imageSegmentationGenerator(images_path, segs_path, batch_size,
                               n_classes, input_height, input_width):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.tif") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.tif") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            im = cv2.imread(im, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(seg, 0)

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] == input_height and im.shape[1] == input_width

            X.append(im)
            seg = seg / 255 if seg.max() > 254 else seg
            Y.append(getSegmentationArr(seg,n_classes,input_height,input_width))
            '''
            X.append(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))
            X.append(cv2.rotate(im, cv2.ROTATE_180))
            X.append(cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE))
            
            Y.append(getSegmentationArr(cv2.rotate(seg, cv2.ROTATE_90_CLOCKWISE),n_classes,input_height,input_width))
            Y.append(getSegmentationArr(cv2.rotate(seg, cv2.ROTATE_180),n_classes,input_height,input_width))
            Y.append(getSegmentationArr(cv2.rotate(seg, cv2.ROTATE_90_COUNTERCLOCKWISE),n_classes,input_height,input_width))
            
            im = cv2.flip(im, 1)
            seg = cv2.flip(seg, 1)
            
            X.append(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))
            X.append(cv2.rotate(im, cv2.ROTATE_180))
            X.append(cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE))
            
            Y.append(cv2.rotate(getSegmentationArr(seg,n_classes,input_height,input_width), cv2.ROTATE_90_CLOCKWISE))
            Y.append(cv2.rotate(getSegmentationArr(seg,n_classes,input_height,input_width), cv2.ROTATE_180))
            Y.append(cv2.rotate(getSegmentationArr(seg,n_classes,input_height,input_width), cv2.ROTATE_90_COUNTERCLOCKWISE))
            '''
        yield np.array(X), np.array(Y)

if __name__ == '__main__':
    G = imageSegmentationGenerator("data/dataset1/images_prepped_train/",
                                   "data/dataset1/annotations_prepped_train/", batch_size=16, n_classes=15, input_height=320, input_width=320)
    x, y = G.__next__()
    print(x.shape, y.shape)
