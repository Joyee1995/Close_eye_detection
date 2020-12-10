import cv2
import time
import numpy as np

eye_cascPath = '/Users/joyee/Documents/C/closed_eye/Closed-Eye-Detection-with-opencv/haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = '/Users/joyee/Documents/C/closed_eye/Closed-Eye-Detection-with-opencv/haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps=6
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width,  height))


fps = []
lasttime = time.time()
while 1:
    ret, img = cap.read()
    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            if len(eyes) == 0:
                cv2.putText(img, "Closed Eyes!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 1, cv2.LINE_AA)
                print('no eyes!!!')
            else:
                print('eyes!!!')
#             frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
            out.write(img)
            cv2.imshow('Face Recognition', img)
        waitkey = cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break

    fps.append(time.time()-lasttime)
    lasttime = time.time()

fps = 1 / np.mean(fps)
print(fps)
print(fps)
print(fps)
print(fps)
print(fps)
cap.release()            
out.release()


