import cv2 as cv
import mediapipe as mp
import time

class HandDetector:
    def __init__(self):
        self.myHands = mp.solutions.hands
        self.hands = self.myHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.myHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lm_list

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])  # print the position of landmark 4

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()