# -*- coding: utf-8 -*-
"""
Created on 12 jan 2018
Project: ImageClassification
File: server
@author: kalyan
"""
import numpy as np
from PIL import Image
from pytesseract import image_to_string
import os, os.path
import cv2
from dateutil.parser import parse
import os
import tensorflow as tf
import re

 

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_path(path):
    if os.path.exists(path):
        return True
    else:
        return False
    
def classify_image():

        f= "test (12).jpg"
        path_ = "C:/Users/"
        path =  path_ + '/' + f
        
        
        if check_path(path):
               directory_path = path_ + '/testoutput/'
               create_directory(directory_path)
               
               image_file_name = path_ + '/' + f
               dest = path + '/processed-images/' + f
               print (image_file_name)
               new_image_name = image_file_name[:-4]
               smooth_image = new_image_name + '_smooth.jpg'
               resize_image = new_image_name + '_resize.jpg'
               thresh_image = new_image_name + '_thresh.jpg'
               Denoise_image = new_image_name + '_denoise.jpg'
               img = cv2.imread(image_file_name)
               #img=cv2.imread("C:/Users/kalyan/kk.jpg")
               
               #Denoise
               dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
               Denoise_img = cv2.imwrite(Denoise_image,dst)
               
               #Smooth
               kernel = np.ones((5,5),np.float32)/25
               dst = cv2.filter2D(dst,-1,kernel)
               smooth_img = cv2.imwrite(smooth_image,dst)
               #Resize  
               im = cv2.imread(Denoise_image)
               res = cv2.resize(im,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
               resize_img = cv2.imwrite(resize_image,res)
               
               #Threshold
               img_thresh = cv2.imread(resize_image,cv2.IMREAD_GRAYSCALE)
               ret,thresh1 = cv2.threshold(img_thresh,127,255,cv2.THRESH_BINARY)
               ret,thresh2 = cv2.threshold(img_thresh,127,255,cv2.THRESH_BINARY_INV)
               ret,thresh3 = cv2.threshold(img_thresh,127,255,cv2.THRESH_TRUNC)
               ret,thresh4 = cv2.threshold(img_thresh,127,255,cv2.THRESH_TOZERO)
               ret,thresh5 = cv2.threshold(img_thresh,127,255,cv2.THRESH_TOZERO_INV)
               
               #Adaptive threshold
               img1 = cv2.imread(image_file_name,0)
               img1 = cv2.medianBlur(img1,5) 
               th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
               th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
               
            
               #Otsu's thresholding
               
               #ret2,th2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

               # Otsu's thresholding after Gaussian filtering
               #blur = cv2.GaussianBlur(img1,(5,5),0)
               #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
               
               titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
               images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
               
               img = cv2.imwrite(thresh_image,thresh3) 
               im_orig = Image.open(image_file_name)
               #im = Image.open(cleaned_image_name_grey)
               img = Image.open(thresh_image)
               text_orig = image_to_string(Image.open(thresh_image), lang='eng')
               print(text_orig)
               
               text_orig = text_orig.encode('ascii', 'ignore').decode('ascii')
               #text = text.decode('unicode_escape').encode('ascii','ignore')
               filename = os.path.splitext(f)[0]
               filename = ''.join(e for e in filename if e.isalnum() or e == '-')
               text_file_path = 'C:/Users/vishnu.jayanand/Desktop/Opencv/test/testoutput/' + filename + '.txt'

               text_file = open(text_file_path, "w+")
               text_file.write("%s" % text_orig)
               text_file.close()
               
               #text_grey = image_to_string(im, lang='eng')
               
               #print text_grey
               

if __name__ == '__main__':
    
    #app.run(host='0.0.0.0')
    classify_image()
