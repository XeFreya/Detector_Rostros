from sys import maxsize
import cv2
import numpy as np

# Le pasamos el script entrenado para la detección de ciertas cosas
# En este caso un rostro
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# imread lee la imagen especifica que tenemos en el proyecto
image = cv2.imread('imagen_de_prueba.jpg')
imageAux = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceClassif.detectMultiScale(gray,
    # Este parametro especifica que tanto es reducida la imagen
    # 1.1 Quiere decir que se va a reducir en un 10 porciento, 1-3
    # En un 30 porciento
    scaleFactor=1.1,
    # Es el numero minimo de vecinos positivos que debe tener
    # una detección correcta para que nos muestre un positivo real
    # y no un falso positivo
    minNeighbors=5,
    # Indica el tamaño minimo del objeto, objeto más pequeño es i
    minSize=(30,30),
    # Indica el tamaño maximo del objeto, objeto más grande es i
    maxSize=(60,60))


count = 0

for (x,y,w,h) in faces:
    #genera un rectangulo al detectar en este caso un rosto
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #Usamos la imagen auxilar para que no nos muestre el recuadro de reconomiento al hacer el recorte
    rostro = imageAux[y:y+h,x:x+w]
    #Redimensionamos la imagen del rostro recortado, no es necesario
    #Usarlo, es solamente para que nos amplie la imagen del recorte, incluso
    #Podemos no usarlo
    rostro = cv2.resize(rostro, (150,120),interpolation=cv2.INTER_CUBIC)
    #Le establecemos un nombre al rostro, usamos 'count' para que sea mas sencillo
    cv2.imwrite(f'rostro_{count}.jpg',rostro)
    count += 1 

    #Visualizamos los rostros recortados, además de la imagen de entrada
    cv2.imshow('rostro',rostro)
    #Muestra la imagen en pantalla
    cv2.imshow('image', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
