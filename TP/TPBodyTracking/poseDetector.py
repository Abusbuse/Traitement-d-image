import cv2 as cv
import mediapipe as mp
import time
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.stage = None
        self.counter = 0
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=1, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        
    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmlist
    
    # Fonction pour calculer l'angle entre 3 points
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        print(angle)
        # Draw
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 0, 0), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    
    def drawAngleBar(self, img, angle):
        bar_length = 200
        bar_thickness = 10
        # Calcul du pourcentage de l'angle
        percentage = (angle - 25) / (180 - 25)
        if percentage > 1:
            percentage = 1
        elif percentage < 0:
            percentage = 0
        filled_length = int(bar_length * percentage)
        bar_x = int((img.shape[1] - bar_length) / 2) 
        bar_y = img.shape[0] - 100 
        cv.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_thickness), (0, 0, 0), cv.FILLED)
        cv.rectangle(img, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_thickness), (0, 255, 0), cv.FILLED)
        
        # Affiche le pourcentage
        cv.putText(img, f"{int(percentage * 100)}%", (bar_x + bar_length + 10, bar_y + bar_thickness), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    def countReps(self, angle):
        if angle > 150:
            self.stage = "up"
        if angle < 100 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter
             
def main():
    cap = cv.VideoCapture('videos/projet.mp4')
    # Rezie the window to 400x300
    cap.set(3, 400)
    cap.set(4, 300)
    
    pTime = 0
    detector = PoseDetector()
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
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()