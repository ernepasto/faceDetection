# Il face detection è uno dei passaggi necessari per effettuare il face recognition, l'approccio basato sul deep learning gestisce il face detection in modo più accurato e veloce rispetto ai metodi tradizionali.

# In questo programma si utilizzana il modello ResNet (basato su VGG) e i sui pesi pre-addestrati forniti dalla comunità OpenCV

# Import
import cv2
import pandas as pd
#import numpy as np

# Impostazione del OpenCV deep neural networks module
detector = cv2.dnn.readNetFromCaffe('deploy.prototxt' , 'res10_300x300_ssd_iter_140000.caffemodel')

# Il modello ResNet SSD si aspetta un'immagine della dimensione di 300x300
image = cv2.imread('./images/foto.jpg')
base_img = image.copy()
original_size = base_img.shape
target_size = (300, 300)
image = cv2.resize(image, target_size)
aspect_ratio_x = (original_size[1] / target_size[1])
aspect_ratio_y = (original_size[0] / target_size[0])
imageBlob = cv2.dnn.blobFromImage(image = image)
#imageBlob = np.expand_dims(np.rollaxis(image, 2, 0), axis = 0)

# Per vedere il risultato sulle immagini
#cv2.imshow('color image', base_img)
#cv2.waitKey(0)
#cv2.imshow('color image', image)
#cv2.waitKey(0)

# Viene passata al modello l'immagine delle corrette dimensioni 
detector.setInput(imageBlob)
detections = detector.forward()

# L'output della rete neurale è una matrice di dimensioni (200, 7)
# Le righe di questa matrice rappresentano i candidati trovati
# Possiamo filtrare i candidati trovati in base alle loro caratteristiche
column_labels = ['img_id', 'is_face', 'confidence', 'left', 'top', 'right', 'bottom']
detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
print('\n\nOutput della rete neurale PRIMA di aver apllicato i filtri:')
print(detections_df)

# La caratteristica is_face sarà 0 per lo sfondo e sarà 1 per i volti, quindi ignoriamo i valori zero
# Ignoriamo anche i casi con un valore di confidenza inferiore a una certa soglia (ad esempio 90%)
# L'applicazione di questi filtri elimina i falsi positivi
# 0: background, 1: face
detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
detections_df = detections_df[detections_df['confidence'] >= 0.65]
detections_df.head()
print('\n\nOutput della rete neurale DOPO di aver apllicato i filtri:')
print(detections_df)

# Estrazione del volto individuato dall'immagine e applicazione del rettangolo sull'immagine con scritte che indicano la confidenza (confidence score) calcolata => valore di match 
for i, instance in detections_df.iterrows():
    
    confidence_score = str(round(100*instance['confidence'], 2))+' %'
    
    left = int(instance['left'] * 300)
    bottom = int(instance['bottom'] * 300)
    right = int(instance['right'] * 300)
    top = int(instance['top'] * 300)
    
    detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

    # Per vedere il risultato sulle immagini
    #cv2.imshow('color image', detected_face)
    #cv2.waitKey(0)
    
    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
        
        cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1)
        
        print('\n\nValori di match score per i volti trovati dalla rete neurale:')
        print('Id ',i)
        print("Confidence: ", confidence_score)

# Per vedere il risultato sulle immagini
cv2.imshow('color image', base_img)
cv2.waitKey(0)
