import cv2 as cv
import time
import handDetector as hd
import numpy as np
import os

# Global variables
overlayList=[]
drawColor = None
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8) # defining canvas

def header(img):
    h, w, c = img.shape
    cv.rectangle(img, (0, 0), (w, 125), (0, 0, 0), cv.FILLED)
    cv.putText(img, "Rouge", (150, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    overlayList.append(img[0:125,0:320])
    cv.putText(img, "Bleu", (450, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    overlayList.append(img[0:125,320:640])
    cv.putText(img, "Vert", (750, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    overlayList.append(img[0:125,640:960])
    cv.putText(img, "Effacer", (1050, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    overlayList.append(img[0:125,960:1280])
    return img

def draw_on_image(img, lmList, detector):
    global xp, yp, drawColor
    if len(lmList)!=0:
        x1, y1 = lmList[8][1],lmList[8][2]
        x2, y2 = lmList[12][1],lmList[12][2]
        fingers = detector.fingersUp()
        
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            if y1 < 125:
                if 0 < x1 < 320:
                    drawColor = (0, 0, 255)
                elif 320 < x1 < 640:
                    drawColor = (255, 0, 0)
                elif 640 < x1 < 960:
                    drawColor = (0, 255, 0)
                elif 960 < x1 < 1280:
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, 50)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 50)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, 20)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 20)

            xp, yp = x1, y1
    return img

def process_image(img):
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)
    return img

def main():
    global imgCanvas, overlayList
    cap=cv.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    detector = hd.handDetector(detectionCon=0.50,maxHands=1)

    while True:
        success, img = cap.read()
        img=cv.flip(img,1)
        
        img = detector.findHands(img)
        lmList,bbox = detector.findPosition(img, draw=False)
        img = header(img)
        img = draw_on_image(img, lmList, detector)
        img = process_image(img)

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
        
if __name__ == "__main__":
    main()