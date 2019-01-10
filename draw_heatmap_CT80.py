from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import pdb
import h5py
import json
import re
import math
import string
import os
import pdb
import PIL
from scipy import interpolate

from PIL import Image, ImageDraw, ImageColor, ImageFont
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join
import scipy.io as sio 
# from __future__ import division

def draw_atten_weights(im_ori, wi, word_l,j):
    wi = (wi - wi.min()) / (wi.max() - wi.min())
    wi_a = np.reshape(wi, (6, 40))
    wi_b = wi_a[:,:word_l]

    # wi_b=wi_a.repeat(8, axis=0).repeat(4, axis=1)
    # wordbg = wi_b[:, 1:161]
    # print(wordbg)
    # pdb.set_trace()
    # y = np.arange(32/4)
    # x = np.arange(160/41)


    # ip = interpolate.interp2d(x,y,wi_a)
    # xn = np.linspace(0,2,160)
    # yn = np.linspace(0,2,32)

    # fnew = ip(xn, yn)
    # print(fnew)

    # wordbg=np.zeros((np.array(wordim).shape[0],np.array(wordim).shape[1]))
    # inter_x= (float(wordbg.shape[1])/word_l)
    # for k in range(1, word_l):
    #     lf=int(round((k-1)*inter))
    #     rt=int(round((k)*inter))
    #     wordbg[:,lf:rt]=wi[k];
    # wd_wi = Image.fromarray(np.uint8(wordbg))
    # wd_wi.save('img2.jpg')
    # pdb.set_trace()
 
    im_wi = Image.fromarray(np.uint8(cm.jet(wi_b)*255))
    im_wi = im_wi.resize(im_ori.size, Image.BICUBIC)
    im_wi_overlay = Image.blend(im_ori.convert("RGBA"), im_wi.convert("RGBA"), 0.5)
    
    draw = ImageDraw.Draw(im_wi_overlay)
    
    filename = 'tmp'+str(j)+'.png'
    im_wi_overlay.save(filename)
    pdb.set_trace()


f1 = h5py.File('Analysis/att_weight_res.h5', 'r')
all_weights=np.array(f1['att_weight'])
with open('vis/results_CT80_BLSTM.json', 'r') as f:
    data = json.load(f)
print(data)
captions = data['captions']

imagePathDir='../Dataset/CUTE80/'
labelfile =open('../Dataset/CUTE80/gt_sensitive.txt')
i=0
while 1:
    line = labelfile.readline()
    if not line:
        break
    i=i+1
    if i==16:
      content=line.split(' ')
      ll=len(content[0])
      label = line[ll+1:-2]
      print(label)

      imagePath=imagePathDir+content[0]
      print(imagePath)
      pdb.set_trace()
      im_ori = Image.open(imagePath)
      hpercent = (48/float(im_ori.size[1]))
      wsize = int(round((float(im_ori.size[0])*float(hpercent))))
      if wsize>160:
        wsize=160
      elif wsize<48:
        wsize=48

      wordim = im_ori.resize((wsize,48), PIL.Image.ANTIALIAS)
      
      # wordim = im_ori.resize((160,32), PIL.Image.ANTIALIAS)
      pw = int(math.floor(wordim.size[0]/4))
      wt_i=all_weights[i-1]
      for j in range(0, 32):
        imag_atten_weights=wt_i[j]

        draw_atten_weights(wordim, imag_atten_weights, pw, j)

