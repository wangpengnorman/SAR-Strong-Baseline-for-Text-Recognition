import os
import glob
import yaml
import lmdb
import cv2
import pdb
import string 
import scipy.io
import numpy as np
import ast
import xml.etree.ElementTree as ET
import pdb

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imagePath
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    FPath=os.getcwd()
    
    for ij in range(1,21):
        imageList=[]
        labelList=[]
        outputPath=FPath+'/DataDB/train_'+str(ij)+'/'   # separate the training data into different groups
        labelfile =open('../Dataset/SynthText_Add/annotationlist/gt_'+str(ij)+'.txt')
        imagePathDir ='../Dataset/SynthText_Add/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            content=line.split(',')
            imagePath=imagePathDir+'crop_img_'+str(ij)+'/'+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][1:-2])
        labelfile.close()

        labelfile = open('../Dataset/SynthText_org/annotationlist/annotation_train_'+str(ij)+'.txt')
        imagePathDir ='../Dataset/SynthText_org/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            content=line.split(',')
            imagePath=imagePathDir+content[0]+'.jpg'
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][:-2])
        labelfile.close()

        labelfile = open('../Dataset/SynthText_org/annotationlist/annotation_train_'+str(ij+99)+'.txt')
        imagePathDir ='../Dataset/SynthText_org/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]+'.jpg'
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][:-2])
        labelfile.close()

        labelfile = open('../Dataset/Max_Syn_90kDICT32px/annotationlist/annotation_train_'+str(ij)+'.txt')
        imagePathDir ='../Dataset/Max_Syn_90kDICT32px'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            filepath_t=line.split(' ')
            filepath=filepath_t[0][1:]
            
            labelp=filepath.split('_')
            labelList.append(labelp[1])
            imagePath=imagePathDir+filepath
            imageList.append(imagePath)
        labelfile.close()

        labelfile = open('../Dataset/Max_Syn_90kDICT32px/annotationlist/annotation_train_'+str(ij+90)+'.txt')
        imagePathDir ='../Dataset/Max_Syn_90kDICT32px'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            filepath_t=line.split(' ')
            filepath=filepath_t[0][1:]
            
            labelp=filepath.split('_')
            # if len(labelp)>2:
            labelList.append(labelp[1])
            imagePath=imagePathDir+filepath
            imageList.append(imagePath)
        labelfile.close()

        labelfile = open('../Dataset/COCO_WordRecognition/train_words_gt.txt')
        imagePathDir ='../Dataset/COCO_WordRecognition/train_words/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]+'.jpg'
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(line[ll+1:-2])
        labelfile.close()

        imagePathDir='../Dataset/IC15Inc_WordRecognition/ch4_training_word_images_gt/'
        labelfile =open('../Dataset/IC15Inc_WordRecognition/ch4_training_word_images_gt/annotation_gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            labelList.append(content[1][2:-2])
        labelfile.close()

        imagePathDir='../Dataset/IC13_WordRecognition/Challenge2_Training_Task3_Images_GT/'
        labelfile =open('../Dataset/IC13_WordRecognition/Challenge2_Training_Task3_Images_GT/train_annotation_gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            labelList.append(content[1][2:-2])
        labelfile.close()

        imagePathDir='../Dataset/IIIT5K/'
        labelfile =open('../Dataset/IIIT5K/train_annotation_gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(line[ll+1:-1])
        labelfile.close()

        labelfile =open('../Dataset/COCO_WordRecognition/val_words_gt.txt')
        imagePathDir='../Dataset/COCO_WordRecognition/val_words/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',')
            imagePath=imagePathDir+content[0]+'.jpg'
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(line[ll+1:-2])
        labelfile.close()

        createDataset(outputPath, imageList, labelList, lexiconList=None, checkValid=True)
