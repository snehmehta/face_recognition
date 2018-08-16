import cv2
from imutils.video import VideoStream
import pickle


face_cascade = cv2.CascadeClassifier('E:\python\opencv\cascades\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("label.pickle",'rb') as f:
    or_labels = pickle.load(f)
    labels = { v:k for k,v in or_labels.items()}

vs = VideoStream(src=0).start()

while True:

    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.5,5)

    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]  

        id_ , conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
        
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame,name,(x,y),font,1,(255,255,255),2,cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    
    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows() 
    