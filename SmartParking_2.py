# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:05:40 2020

@author: Sameed Khan Durrani
"""

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import SmartParking_2_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    SmartParking_2_support.init(root, top)
    root.mainloop()
    

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    SmartParking_2_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None
    
    
    
    # -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:05:40 2020

@author: Sameed Khan Durrani
"""
import tensorflow as tf



import cv2
import sys

import numpy as np
from keras.preprocessing import image
#from tkinter import PhotoImage
from tkinter import *
from PIL import ImageTk,Image


class Toplevel1:
    
    model = tf.keras.models.Sequential()
    model2 = tf.keras.models.Sequential()
    
    def importmodel(self):
        global model
        global model2
        json_file = open('E:/page/Project/model_car_finalized.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model =  tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("E:/page/Project/model_car_finalized.h5")
        print("Loaded model from disk")
        
        
        json_file = open('E:/page/Project/modelplate.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model2 =  tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model2.load_weights("E:/page/Project/modelplateweights.h5")
        print("Loaded model from disk")
        return


    def detectcarandplate(self):
        if 1==1:
          path=  "C:/Users/Sameed Khan Durrani/Desktop/CARPLATES/pakroisay3.jfif"
          img = cv2.imread(path)
          origimg=img.copy()
        
          cv2.setUseOptimized(True)
          cv2.setNumThreads(8)
        
          gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
          gs.setBaseImage(img)
        
          #gs.switchToSingleStrategy()
          #gs.switchToSelectiveSearchFast()
          gs.switchToSelectiveSearchQuality()
        
        
          rects = gs.process()
          nb_rects = 150
        
          wimg = img.copy()
          croppedimages=[]
          h_img, w_img, c_img = wimg.shape
          
          for i in range(len(rects)):
                      if (i < nb_rects):
                          x, y, w, h = rects[i]
                          if h>w:
                            if h/w>1.2 and h/w<1.8 and h>(h_img/6) and w>(w_img/6):
                          #    cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                              croppedimages.append(wimg[y:y+h, x:x+w])
                          else:
                            if w/h>1.2 and w/h<1.8 and h>(h_img/6) and w>(w_img/6):
                          #    cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                              croppedimages.append(wimg[y:y+h, x:x+w])
        
#          cv2.imshow("wimg", wimg);
#          c = cv2.waitKey()
        
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
              #cv2.imshow( "imgcropped", eachCropped);
              c = cv2.waitKey()
              imgs.append(eachCropped)
        
        
          cv2.destroyAllWindows()
          
          
        
        for imj in imgs:
                    
          cv2.imwrite('tsvr.png',imj)
          imj=cv2.imread('tsvr.png')
          
        #  cv2_imshow( img);
        
        #  cv2.setUseOptimized(True)
        #  cv2.setNumThreads(8)
        
        #  gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
          gs.setBaseImage(imj)
        
          #gs.switchToSingleStrategy()
          #gs.switchToSelectiveSearchFast()
        #  gs.switchToSelectiveSearchQuality()
        
        
          rects = gs.process()
          nb_rects = 50
        
          wimg = imj.copy()
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
                    classes = model2.predict(images, batch_size=10)
                    #print(classes[0])
                    
                    #cv2.imshow( "imgcropped", eachCropped);
                    c = cv2.waitKey()
            
                    if classes[0]>0.5:
                      print("It is a plate")
                      #cv2.imshow( "imgcropped", eachCropped);
                      #c = cv2.waitKey()
                      #cv2.imshow( "Car", imj);
                      #c = cv2.waitKey()
                              
                      cv2.imwrite('imjj2.png',cv2.resize(eachCropped, (150, 100))  )
                      cv2.imwrite('imjjj.png',cv2.resize(imj, (150, 150)))  
                      cv2.imwrite('origimj.png',cv2.resize(origimg, (150, 150)))
                                  
                      origimj = ImageTk.PhotoImage(Image.open("origimj.png")) 
                      imjj = ImageTk.PhotoImage(Image.open("imjjj.png"))  
                      #imj2= ImageTk.PhotoImage(Image.open("imjj2.png"))  
                      #self.Canvas1.pack()
                      #self.Canvas2.pack()
                      self.Canvas1.create_image(10, 10, image=origimj, anchor=NW)
                      self.Canvas12.create_image(10, 10, image=imjj, anchor=NW)
                      #self.Canvas2.create_image(10, 10, image=imj2, anchor=NW)
                      print("set on canvas")
                      #self.mainloop()
                      self.mainloop()
                      break
                else:
                    continue
                    #else:
                    #  print(fn + " is a negative")
                cv2.destroyAllWindows()
        print('done')
                
        return
    
    def showplate(self):
        
        imj2= ImageTk.PhotoImage(Image.open("imjj2.png"))  
        self.Canvas2.create_image(10, 10, image=imj2, anchor=NW)
        self.mainloop()
        return

    
    def putchars(self):
        imgss=[]
        
        img0 = ImageTk.PhotoImage(Image.open("0.png")) 
        self.Canvas11.create_image(0, 0, image=img0, anchor=NW)
        img1 = ImageTk.PhotoImage(Image.open("1.png")) 
        self.Canvas3.create_image(0, 0, image=img1, anchor=NW)
        img2 = ImageTk.PhotoImage(Image.open("2.png")) 
        self.Canvas4.create_image(0, 0, image=img2, anchor=NW)
        img3 = ImageTk.PhotoImage(Image.open("3.png")) 
        self.Canvas5.create_image(0, 0, image=img3, anchor=NW)
        img4 = ImageTk.PhotoImage(Image.open("4 (1).png")) 
        self.Canvas6.create_image(0, 0, image=img4, anchor=NW)
        img5 = ImageTk.PhotoImage(Image.open("5.png")) 
        self.Canvas7.create_image(0, 0, image=img5, anchor=NW)
        img6 = ImageTk.PhotoImage(Image.open("6.png")) 
        self.Canvas8.create_image(0, 0, image=img6, anchor=NW)
        img7 = ImageTk.PhotoImage(Image.open("7.png")) 
        self.Canvas9.create_image(0, 0, image=img7, anchor=NW)
        img8 = ImageTk.PhotoImage(Image.open("8.png")) 
        self.Canvas10.create_image(0, 0, image=img8, anchor=NW)
        self.mainloop()
        return
    
    
    def showpath(self):
                      
        imgpath = ImageTk.PhotoImage(Image.open("path.png")) 
        self.Canvas11.create_image(0, 0, image=imgpath, anchor=NW)
        self.mainloop()
        return

    def __init__(self, top=None):
            
    
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'

        top.geometry("1162x651+-5+0")
        top.minsize(120, 1)
        top.maxsize(1370, 749)
        top.resizable(1, 1)
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.Canvas1 = tk.Canvas(top)
        self.Canvas1.place(relx=0.0, rely=0.0, relheight=0.458, relwidth=0.356)
        self.Canvas1.configure(background="#d9d9d9")
        self.Canvas1.configure(borderwidth="2")
        self.Canvas1.configure(highlightbackground="#d9d9d9")
        self.Canvas1.configure(highlightcolor="black")
        self.Canvas1.configure(insertbackground="black")
        self.Canvas1.configure(relief="ridge")
        self.Canvas1.configure(selectbackground="#c4c4c4")
        self.Canvas1.configure(selectforeground="black")


        
        self.ImportModel = tk.Button(top, text='''ImportModel''', command = self.importmodel)
        self.ImportModel.place(relx=0.809, rely=0.015, height=34, width=177)
        self.ImportModel.configure(activebackground="#ececec")
        self.ImportModel.configure(activeforeground="#000000")
        self.ImportModel.configure(background="#d9d9d9")
        self.ImportModel.configure(disabledforeground="#a3a3a3")
        self.ImportModel.configure(foreground="#000000")
        self.ImportModel.configure(highlightbackground="#d9d9d9")
        self.ImportModel.configure(highlightcolor="black")
        self.ImportModel.configure(pady="0")

        self.Canvas2 = tk.Canvas(top)
        self.Canvas2.place(relx=0.361, rely=0.0, relheight=0.184, relwidth=0.373)

        self.Canvas2.configure(background="#d9d9d9")
        self.Canvas2.configure(borderwidth="2")
        self.Canvas2.configure(highlightbackground="#d9d9d9")
        self.Canvas2.configure(highlightcolor="black")
        self.Canvas2.configure(insertbackground="black")
        self.Canvas2.configure(relief="ridge")
        self.Canvas2.configure(selectbackground="#c4c4c4")
        self.Canvas2.configure(selectforeground="black")

        self.Canvas3 = tk.Canvas(top)
        self.Canvas3.place(relx=0.361, rely=0.184, relheight=0.095
                , relwidth=0.09)
        self.Canvas3.configure(background="#d9d9d9")
        self.Canvas3.configure(borderwidth="2")
        self.Canvas3.configure(highlightbackground="#d9d9d9")
        self.Canvas3.configure(highlightcolor="black")
        self.Canvas3.configure(insertbackground="black")
        self.Canvas3.configure(relief="ridge")
        self.Canvas3.configure(selectbackground="#c4c4c4")
        self.Canvas3.configure(selectforeground="black")

        self.Canvas4 = tk.Canvas(top)
        self.Canvas4.place(relx=0.456, rely=0.184, relheight=0.095
                , relwidth=0.088)
        self.Canvas4.configure(background="#d9d9d9")
        self.Canvas4.configure(borderwidth="2")
        self.Canvas4.configure(highlightbackground="#d9d9d9")
        self.Canvas4.configure(highlightcolor="black")
        self.Canvas4.configure(insertbackground="black")
        self.Canvas4.configure(relief="ridge")
        self.Canvas4.configure(selectbackground="#c4c4c4")
        self.Canvas4.configure(selectforeground="black")

        self.Canvas5 = tk.Canvas(top)
        self.Canvas5.place(relx=0.551, rely=0.184, relheight=0.095
                , relwidth=0.09)
        self.Canvas5.configure(background="#d9d9d9")
        self.Canvas5.configure(borderwidth="2")
        self.Canvas5.configure(highlightbackground="#d9d9d9")
        self.Canvas5.configure(highlightcolor="black")
        self.Canvas5.configure(insertbackground="black")
        self.Canvas5.configure(relief="ridge")
        self.Canvas5.configure(selectbackground="#c4c4c4")
        self.Canvas5.configure(selectforeground="black")

        self.Canvas6 = tk.Canvas(top)
        self.Canvas6.place(relx=0.645, rely=0.184, relheight=0.095
                , relwidth=0.09)
        self.Canvas6.configure(background="#d9d9d9")
        self.Canvas6.configure(borderwidth="2")
        self.Canvas6.configure(highlightbackground="#d9d9d9")
        self.Canvas6.configure(highlightcolor="black")
        self.Canvas6.configure(insertbackground="black")
        self.Canvas6.configure(relief="ridge")
        self.Canvas6.configure(selectbackground="#c4c4c4")
        self.Canvas6.configure(selectforeground="black")

        self.Canvas7 = tk.Canvas(top)
        self.Canvas7.place(relx=0.361, rely=0.292, relheight=0.097
                , relwidth=0.09)
        self.Canvas7.configure(background="#d9d9d9")
        self.Canvas7.configure(borderwidth="2")
        self.Canvas7.configure(highlightbackground="#d9d9d9")
        self.Canvas7.configure(highlightcolor="black")
        self.Canvas7.configure(insertbackground="black")
        self.Canvas7.configure(relief="ridge")
        self.Canvas7.configure(selectbackground="#c4c4c4")
        self.Canvas7.configure(selectforeground="black")

        self.Canvas8 = tk.Canvas(top)
        self.Canvas8.place(relx=0.456, rely=0.292, relheight=0.097
                , relwidth=0.088)
        self.Canvas8.configure(background="#d9d9d9")
        self.Canvas8.configure(borderwidth="2")
        self.Canvas8.configure(highlightbackground="#d9d9d9")
        self.Canvas8.configure(highlightcolor="black")
        self.Canvas8.configure(insertbackground="black")
        self.Canvas8.configure(relief="ridge")
        self.Canvas8.configure(selectbackground="#c4c4c4")
        self.Canvas8.configure(selectforeground="black")

        self.Canvas9 = tk.Canvas(top)
        self.Canvas9.place(relx=0.551, rely=0.292, relheight=0.097
                , relwidth=0.09)
        self.Canvas9.configure(background="#d9d9d9")
        self.Canvas9.configure(borderwidth="2")
        self.Canvas9.configure(highlightbackground="#d9d9d9")
        self.Canvas9.configure(highlightcolor="black")
        self.Canvas9.configure(insertbackground="black")
        self.Canvas9.configure(relief="ridge")
        self.Canvas9.configure(selectbackground="#c4c4c4")
        self.Canvas9.configure(selectforeground="black")

        self.Canvas10 = tk.Canvas(top)
        self.Canvas10.place(relx=0.645, rely=0.292, relheight=0.097
                , relwidth=0.09)
        self.Canvas10.configure(background="#d9d9d9")
        self.Canvas10.configure(borderwidth="2")
        self.Canvas10.configure(highlightbackground="#d9d9d9")
        self.Canvas10.configure(highlightcolor="black")
        self.Canvas10.configure(insertbackground="black")
        self.Canvas10.configure(relief="ridge")
        self.Canvas10.configure(selectbackground="#c4c4c4")
        self.Canvas10.configure(selectforeground="black")

        self.Text1 = tk.Text(top)
        self.Text1.place(relx=0.361, rely=0.399, relheight=0.052, relwidth=0.373)

        self.Text1.configure(background="white")
        self.Text1.configure(font="TkTextFont")
        self.Text1.configure(foreground="black")
        self.Text1.configure(highlightbackground="#d9d9d9")
        self.Text1.configure(highlightcolor="black")
        self.Text1.configure(insertbackground="black")
        self.Text1.configure(selectbackground="#c4c4c4")
        self.Text1.configure(selectforeground="black")
        self.Text1.configure(wrap="word")

        self.Canvas11 = tk.Canvas(top)
        self.Canvas11.place(relx=0.361, rely=0.476, relheight=0.435
                , relwidth=0.381)
        self.Canvas11.configure(background="#d9d9d9")
        self.Canvas11.configure(borderwidth="2")
        self.Canvas11.configure(highlightbackground="#d9d9d9")
        self.Canvas11.configure(highlightcolor="black")
        self.Canvas11.configure(insertbackground="black")
        self.Canvas11.configure(relief="ridge")
        self.Canvas11.configure(selectbackground="#c4c4c4")
        self.Canvas11.configure(selectforeground="black")

        self.Canvas12 = tk.Canvas(top)
        self.Canvas12.place(relx=0.0, rely=0.476, relheight=0.435
                , relwidth=0.355)
        self.Canvas12.configure(background="#d9d9d9")
        self.Canvas12.configure(borderwidth="2")
        self.Canvas12.configure(highlightbackground="#d9d9d9")
        self.Canvas12.configure(highlightcolor="black")
        self.Canvas12.configure(insertbackground="black")
        self.Canvas12.configure(relief="ridge")
        self.Canvas12.configure(selectbackground="#c4c4c4")
        self.Canvas12.configure(selectforeground="black")

        self.ImportImage = tk.Button(top, text='''ImportImage''')
        self.ImportImage.place(relx=0.809, rely=0.154, height=34, width=177)
        self.ImportImage.configure(activebackground="#ececec")
        self.ImportImage.configure(activeforeground="#000000")
        self.ImportImage.configure(background="#d9d9d9")
        self.ImportImage.configure(disabledforeground="#a3a3a3")
        self.ImportImage.configure(foreground="#000000")
        self.ImportImage.configure(highlightbackground="#d9d9d9")
        self.ImportImage.configure(highlightcolor="black")
        self.ImportImage.configure(pady="0")
    
        self.DetectCar = tk.Button(top, text='''DetectCar''', command= lambda: self.detectcarandplate())
        self.DetectCar.place(relx=0.809, rely=0.292, height=34, width=177)
        self.DetectCar.configure(activebackground="#ececec")
        self.DetectCar.configure(activeforeground="#000000")
        self.DetectCar.configure(background="#d9d9d9")
        self.DetectCar.configure(disabledforeground="#a3a3a3")
        self.DetectCar.configure(foreground="#000000")
        self.DetectCar.configure(highlightbackground="#d9d9d9")
        self.DetectCar.configure(highlightcolor="black")
        self.DetectCar.configure(pady="0")

        self.ExtractPlate = tk.Button(top, text='''ExtractPlate''', command= lambda: self.showplate())
        self.ExtractPlate.place(relx=0.809, rely=0.43, height=34, width=177)
        self.ExtractPlate.configure(activebackground="#ececec")
        self.ExtractPlate.configure(activeforeground="#000000")
        self.ExtractPlate.configure(background="#d9d9d9")
        self.ExtractPlate.configure(disabledforeground="#a3a3a3")
        self.ExtractPlate.configure(foreground="#000000")
        self.ExtractPlate.configure(highlightbackground="#d9d9d9")
        self.ExtractPlate.configure(highlightcolor="black")
        self.ExtractPlate.configure(pady="0")
        self.ExtractPlate.configure(text='''ExtractPlate''')

        self.ReadPlate = tk.Button(top, text='''ReadPlate''', command= lambda: self.putchars())
        self.ReadPlate.place(relx=0.809, rely=0.568, height=34, width=177)
        self.ReadPlate.configure(activebackground="#ececec")
        self.ReadPlate.configure(activeforeground="#000000")
        self.ReadPlate.configure(background="#d9d9d9")
        self.ReadPlate.configure(disabledforeground="#a3a3a3")
        self.ReadPlate.configure(foreground="#000000")
        self.ReadPlate.configure(highlightbackground="#d9d9d9")
        self.ReadPlate.configure(highlightcolor="black")
        self.ReadPlate.configure(pady="0")
        self.ReadPlate.configure(text='''ReadPlate''')

        self.FindSlot = tk.Button(top)
        self.FindSlot.place(relx=0.809, rely=0.707, height=34, width=177)
        self.FindSlot.configure(activebackground="#ececec")
        self.FindSlot.configure(activeforeground="#000000")
        self.FindSlot.configure(background="#d9d9d9")
        self.FindSlot.configure(disabledforeground="#a3a3a3")
        self.FindSlot.configure(foreground="#000000")
        self.FindSlot.configure(highlightbackground="#d9d9d9")
        self.FindSlot.configure(highlightcolor="black")
        self.FindSlot.configure(pady="0")
        self.FindSlot.configure(text='''FindSlot''')

        self.ShowPath = tk.Button(top, text='''ShowPath''', command= lambda: self.showpath())
        self.ShowPath.place(relx=0.809, rely=0.845, height=34, width=177)
        self.ShowPath.configure(activebackground="#ececec")
        self.ShowPath.configure(activeforeground="#000000")
        self.ShowPath.configure(background="#d9d9d9")
        self.ShowPath.configure(disabledforeground="#a3a3a3")
        self.ShowPath.configure(foreground="#000000")
        self.ShowPath.configure(highlightbackground="#d9d9d9")
        self.ShowPath.configure(highlightcolor="black")
        self.ShowPath.configure(pady="0")
        self.ShowPath.configure(text='''ShowPath''')

if __name__ == '__main__':
    vp_start_gui()





