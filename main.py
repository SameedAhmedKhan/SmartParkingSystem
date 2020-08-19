# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:05:40 2020

@author: Sameed Khan Durrani
"""
import tensorflow as tf

json_file = open('E:/page/Project/model_car_finalized.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model =  tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("E:/page/Project/model_car_finalized.h5")
print("Loaded model from disk")



import cv2
import sys

import numpy as np
from keras.preprocessing import image

if 1==1:
  path=  "C:/Users/Sameed Khan Durrani/Desktop/CARPLATES/download (3).jfif"
  img = cv2.imread(path)

  cv2.setUseOptimized(True)
  cv2.setNumThreads(8)

  gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
  gs.setBaseImage(img)

  #gs.switchToSingleStrategy()
  #gs.switchToSelectiveSearchFast()
  gs.switchToSelectiveSearchQuality()


  rects = gs.process()
  nb_rects = 80

  wimg = img.copy()
  croppedimages=[]
  h_img, w_img, c_img = wimg.shape
  
  for i in range(len(rects)):
              if (i < nb_rects):
                  x, y, w, h = rects[i]
                  if h>w:
                    if h/w>1.2 and h/w<1.8 and h>(h_img/10) and w>(w_img/10):
                      cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                      croppedimages.append(wimg[y:y+h, x:x+w])
                  else:
                    if w/h>1.2 and w/h<1.8 and h>(h_img/10) and w>(w_img/10):
                      cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                      croppedimages.append(wimg[y:y+h, x:x+w])

  #cv2.imshow("wimg", wimg);
  c = cv2.waitKey()

  imgs=[]
  for eachCropped in croppedimages:

    img = cv2.resize(eachCropped, (150, 150)) 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    classes_prob = model.predict_proba(images, batch_size=1)
    print(classes[0])
    print(classes_prob[0])
    
    if classes[0]>0.5:
      print(path+" is a road")
    else:
      print(path + " is a car")
      cv2.imshow( "imgcropped", eachCropped);
      c = cv2.waitKey()
      imgs.append(eachCropped)


  cv2.destroyAllWindows()
  
  

json_file = open('E:/page/Project/modelplate.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model =  tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("E:/page/Project/modelplateweights.h5")
print("Loaded model from disk")

for img in imgs:
  
#  cv2_imshow( img);

#  cv2.setUseOptimized(True)
#  cv2.setNumThreads(8)

#  gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
  gs.setBaseImage(img)

  #gs.switchToSingleStrategy()
  #gs.switchToSelectiveSearchFast()
#  gs.switchToSelectiveSearchQuality()


  rects = gs.process()
  nb_rects = 50

  wimg = img.copy()
  croppedimages=[]

  for i in range(len(rects)):
              if (i < nb_rects):
                  x, y, w, h = rects[i]
  #                cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                  if w/h>1.3 and w/h <1.7:
                   croppedimages.append(wimg[y:y+h, x:x+w])
                  

 # cv2_imshow( wimg);
  c = cv2.waitKey()

  print(len(croppedimages))
  if len(croppedimages)>=1:
      
      for eachCropped in croppedimages:
        print(eachCropped.shape)
        c = cv2.waitKey()
    
        #cv2_imshow( eachCropped);
        if eachCropped.shape[0]>1 and eachCropped.shape[1]>1:
            img = cv2.resize(eachCropped, (150, 150)) 
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
        
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            #print(classes[0])
            
            #cv2.imshow( "imgcropped", eachCropped);
            c = cv2.waitKey()
    
            if classes[0]>0.5:
              print("It is a plate")
              cv2.imshow( "imgcropped", eachCropped);
              c = cv2.waitKey()
              break
        else:
            continue
            #else:
            #  print(fn + " is a negative")
        
        cv2.destroyAllWindows()