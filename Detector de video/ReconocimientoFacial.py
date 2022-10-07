import cv2
import os
dataPath = 'C:/Users/josem/Desktop/Detector de Rostros/Detector de video/Data'
imagePaths = os.listdir(dataPath)
print("imagePaths", imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Leyendo el modelo
face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resized(rostro, interpolation=cv.INTER_NEAREST)
        result = face_recognizer.predict(rostro)
        
        cv2.putText(frame)