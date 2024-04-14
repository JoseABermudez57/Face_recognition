import cv2
import os
import numpy as np

dataPath = 'Images'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir

    for fileName in os.listdir(personPath):
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
    label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Training...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('AlfredoFaceModel.xml')

print("Modelo almacenado")