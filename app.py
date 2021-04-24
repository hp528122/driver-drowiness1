import flask
from flask import Flask, redirect, url_for , request, render_template
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import numpy as np
from pygame import mixer
import time
from keras import backend as K

def drowsinessdetector():
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
    
    lbl = ['Close','Open']
    
    model = load_model('models/drowsiness_detection.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]
    
    while(True):
        ret, frame = cap.read()
        height, width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)

        for (x,y,w,h) in right_eye:
            r_eye = frame[y:y+h,x:x+w]
            count = count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye = r_eye/255
            r_eye =  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0] == 1):
                lbl='Open' 
            if(rpred[0] == 0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye = frame[y:y+h,x:x+w]
            count = count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0] == 1):
                lbl='Open'   
            if(lpred[0] == 0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score = score+1
            cv2.putText(frame, "Closed", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
        else:
            score = score-1
            cv2.putText(frame, "Open", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

        if(score<0):
            score = 0   
        cv2.putText(frame, 'Score:' + str(score), (100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
        if(score>15):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'), frame)
            try:
                sound.play()

            except:  #isplaying = False
                pass
            if(thicc<16):
                thicc = thicc+2
            else:
                thicc = thicc-2
                if(thicc<2):
                    thicc = 2
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc) 
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

app = flask.Flask(__name__) 
app.debug = True

@app.route("/",methods=['GET', 'POST'])
def home():
    return render_template("index.html")
    
@app.route("/test1",methods=['GET', 'POST'])
def test1():
    return render_template("test1.html")
    
@app.route("/start", methods=['GET', 'POST'])
def start():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Start') == 'Start':
            drowsinessdetector()
            return render_template("test1.html")
    else:
        return render_template("test1.html")

@app.route("/contact",methods=['GET', 'POST'])
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=False)