import time
import numpy as np
import cvlib as cv
import cv2

cap = cv2.VideoCapture(0)

while True:
    _ , frame = cap.read()

    startFrame = time.time()

    face , confidence = cv.detect_face(frame)

    for idx , f in enumerate(face):
        (startX , startY) = f[0] , f[1]
        (endX, endY) = f[2] , f[3]

        cv2.rectangle(frame , (startX , startY) , (endX , endY) , (0,255,0) , 2)
        # text = str(confidence[idx] * 100)[:5]
        # cv2.putText(frame , text , (startX , startY-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (0,255,0) , 2)

        face_crop = np.copy(frame[startY:endY, startX:endX])
        label , confidence = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        label = label[idx]
        text = label + "( " + str(confidence[idx]*100)[:5] + ")"
        cv2.putText(frame , text , (startX , startY+20) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (0,0,255) , 2)


    cv2.imshow("Gender Detection" , frame)
    # print(time.time() - startFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()