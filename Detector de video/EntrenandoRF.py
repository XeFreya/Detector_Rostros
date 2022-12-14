import cv2
import os
import numpy as np

dataPath = 'C:/Users/josem/Desktop/Detector de Rostros/Detector de video/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName, 0)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    label += 1
    
print('labels= ',labels)
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenando el reconocedor de Rostros
print("Entrenando. . .")
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')

print('Modelo almacenado')


    
        