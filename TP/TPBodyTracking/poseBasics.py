import cv2 as cv
import mediapipe as mp
import time

def main():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    
    # cap = cv.VideoCapture(0)
    # cap = cv.VideoCapture('videos/FV_Sprint.avi')
    # cap = cv.VideoCapture('videos/RaceWalking.avi')
    # cap = cv.VideoCapture('videos/Rowing.avi')
    cap = cv.VideoCapture('videos/bras.mp4')
    pTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()