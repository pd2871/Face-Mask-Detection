"""
** Important **

Download the model.h5 file from given link below and save it in required directory

https://drive.google.com/file/d/1-xYqzzyI2jukKlarfmns0VuQVJBs3stj/view?usp=sharing

"""
import cv2,mtcnn
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
classifier = load_model('model.h5') #Classifier for predicting mask or not mask


#Initializing the front camera
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read() #It read the frames from the live camera
    fcs=[]
    preds=[]
    locs=[]
    cfd=[]

    """
    Now we will use either of Haar Cascade or mtcnn to detect the faces through 
    bounding boxes and use classifier to predict the faces
    """

    """
    ***Note***
    You can choose any one of the Haar Cascade Classifier or mtcnn. If you choose one, 
    don't forget to comment the code of another one.
    """

    # # --For Haar Cascade Classifier--
    # """
    # It is faster than mtcnn but less accurate to detect variety of faces
    # """
    # hc = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # hcfaces = hc.detectMultiScale(frame,1.2)
    # for (x,y,w,h) in hcfaces:
    #     x2=x+w
    #     y2=y+h
    #     fc=frame[y:y2, x:x2] #capturing only faces from the frame
    #     fc = cv2.resize(fc, (150, 150)) # resizing the faces for prediction
    #     fcs.append(fc)
    #     locs.append((x, y, x2, y2))
    # # --End of Haar Cascade Classifier code--


    # --For mtcnn--
    """
    (MTCNN requires large computational power, but gives better accuracy for face detection than
    Haar Cascade. This model is not recommended for computers having lower computing power)
    """
    mt = MTCNN()
    faces = mt.detect_faces(frame) #it returns dictionary containing bbox, confidence and
    # keypoints of eyes,nose,mouth

    for face in faces:
        x,y,w,h = face['box']
        confidence = face['confidence']
        x2=x+w
        y2=y+h
        fc=frame[y:y2, x:x2]
        fc= cv2.resize(fc, (150,150))
        fcs.append(fc)
        locs.append((x, y, x2, y2))
        cfd.append(confidence)

    # --End of mtcnn code--


    conf=0
    if len(fcs) > 0:
        fcs = np.array(fcs, dtype="float32") # Just ensuring that images are in array form
        preds = classifier.predict(fcs, batch_size=32)

    for (bbox, pred) in zip(locs, preds):
        (x, y, x2, y2) = bbox
        label = "Mask" if pred[0]>0.5 else "No Mask"
        if label == "Mask":
            conf= pred[0] * 100 # Confidence % of mask
        else:
            conf = (100 - pred[0] * 100) #Confidence% for non-mask

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # green if mask else red
        label = "{}: {:.2f}%".format(label, conf) # % of confidence


        cv2.rectangle(frame, (x, y), (x2, y2), color, 3)  # Putting rectangle of bbox in frames
        cv2.rectangle(frame, (x,y-40), (x2,y), color, -1)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Putting text in live frames
        cv2.imshow('frame', frame)


    if (cv2.waitKey(20) == ord('q')) or (cv2.waitKey(20) == 27):
        # pressing 'q' or 'esc' keys destroys the window
        break

cap.release()
cv2.destroyAllWindows()

"""
The accuracy of the classifier can be increased by changing it with a new, more accurate classifier
or fine tuning the current one again
"""