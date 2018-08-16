import cv2 
import os 
import numpy as np 
from PIL import Image
import pickle 

Base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(Base_dir,'image')

face_cascade = cv2.CascadeClassifier('E:\python\opencv\cascades\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_label = []
x_train = []

current_id = 0
label_ids = {}

for root,dirname,filename in os.walk(image_path):
    for file in filename:
        path = os.path.join(root,file)
        label = os.path.basename(root).replace(" ","-").lower()
        # print(label,path)

        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        
        id_ = label_ids[label]
        
        pil_image = Image.open(path).convert('L')
        image_array = np.array(pil_image,dtype='uint8')
        
        faces = face_cascade.detectMultiScale(image_array,1.5,5)
        
        for (x,y,w,h) in faces:
            roi = image_array[y:y+h,x:x+w]
            x_train.append(roi)
            y_label.append(id_)

with open("label.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_label))
recognizer.save("trainner.yml")