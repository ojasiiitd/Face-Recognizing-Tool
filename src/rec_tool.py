import os
import time
import cv2
import numpy as np
import face_recognition

REC_DIR = "recognized/"
TOLERANCE = 0.7
MODEL = "hog"

cap = cv2.VideoCapture(0)

face_map_names = []
face_map_encodings = []

for name in os.listdir(REC_DIR):
    image = face_recognition.load_image_file(REC_DIR+name)
    encoding = face_recognition.face_encodings(image)[0]
    face_map_names.append(name)
    face_map_encodings.append(encoding)
print("Sucessfully processed all known faces.\n")

while True:
    _ , frame = cap.read()
    startFrame = time.time()

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    face_loc = face_recognition.face_locations(gray , model=MODEL)
    if len(face_loc) == 0 :
        continue

    face_enc = face_recognition.face_encodings(frame , model="small")

    recognized = False
    for cur_encoding , location in zip(face_enc , face_loc):
        results = face_recognition.compare_faces(face_map_encodings , cur_encoding , TOLERANCE)

        for res in range(len(results)):
            if res == True:
                recognized = True
                match = face_map_names[res]

                top_left = (location[3] , location[0])
                bot_right = (location[1] , location[2])

                cv2.rectangle(frame , top_left , bot_right , (0,255,0) , 2)
                cv2.rectangle(frame , (top_left[0] , top_left[1]-20) , (bot_right[0] , top_left[1]) , (0,255,0) , cv2.FILLED)
                cv2.putText(frame , match , (top_left[0] , top_left[1]-5) , cv2.FONT_HERSHEY_PLAIN , 1 , (0,0,255) , 2)
                cv2.imshow("Frame" , frame)

    if not recognized:
        print("Press ANY KEY to identify the new face.")
        cv2.imshow("New Face" , frame)
        cv2.waitKey(0)
        newName = input("New face detected! Provide a name ('N' if it is not a new face): ")
        print()
        if newName == 'N':
            continue
        cv2.destroyWindow("New Face")
        face_map_names.append(newName)
        face_map_encodings.append(face_enc[0])
        cv2.imwrite(REC_DIR + newName + ".jpg" , frame[face_loc[0][0]-60:face_loc[0][2]+60 , face_loc[0][3]-60:face_loc[0][1]+60])

    print("Time taken per frame:" , str(time.time() - startFrame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()