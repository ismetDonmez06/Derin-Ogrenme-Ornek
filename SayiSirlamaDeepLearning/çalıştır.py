import cv2
import pickle
import numpy as np

def preProcces(img):
    img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =cv2.equalizeHist(img)
    img =img/255

    return img


cap=cv2.VideoCapture(1)
cap.set(3,480)
cap.set(4,480)
pickle_in=open("model_train.p","rb")
model =pickle.load(pickle_in)

while True:
    succes ,frame =cap.read()
    img =np.asarray(frame)
    img=cv2.resize(img,(32,32))
    img =preProcces(img)

    img=img.reshape(1,32,32,1)


    #predict

    classIndex=int(model.predict_classes(img))
    predecitons=model.predict(img)

    probVal =np.amax(predecitons)
    print(classIndex,probVal)

    if probVal >0.7:
        cv2.putText(frame,str(classIndex)+" " + str(probVal), (50,50) ,cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)

    cv2.imshow("Sınıflandırma",frame)
    if cv2.waitKey(1) & 0xff ==ord("q") :break

