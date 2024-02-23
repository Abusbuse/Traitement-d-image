import cv2 as cv
import mediapipe as mp
import time
import math
from poseDetector import PoseDetector

def setup_capture(width=400, height=300):
    cap = cv.VideoCapture('videos/projet.mp4')
    cap.set(3, width)
    cap.set(4, height)
    return cap

def calculate_fps(pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    return fps, cTime

def draw_fps(img, fps):
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)


def main():
    cap = setup_capture()
    pTime = 0
    detector = PoseDetector()
    
    last_angle = 0
    rep_count = 0

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img, draw=False)

        if len(lmlist) != 0:
            cv.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0, 0, 255), cv.FILLED)
            detector.lmlist = lmlist
            angle = detector.findAngle(img, 23, 25, 27)
            rep_count = detector.countReps(angle)
            # Affiche le compteur de répétitions sur l'image
            cv.putText(img, f"Reps: {rep_count}", (70, 100), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            
        # Affiche une barre 
        detector.drawAngleBar(img, angle)
        
        fps, pTime = calculate_fps(pTime)
        draw_fps(img, fps)

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()