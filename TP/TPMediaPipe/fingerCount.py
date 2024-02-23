import cv2 as cv
import time
import os
from handDetector import HandDetector

def load_images(path):
    images = [None]
    filenames = sorted(os.listdir(path), key=lambda filename: int(filename.split('.')[0]))
    for filename in filenames:
        image = cv.imread(os.path.join(path, filename))
        images.append(image)
    return images

def count_fingers(lm_list):
    tipIds = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if lm_list[tipIds[0]][1] > lm_list[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):  # y axis
        if lm_list[tipIds[id]][2] < lm_list[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = fingers.count(1)
    return total_fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()

    path = "./Fingers"
    overlay_images = load_images(path)

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            total_fingers = count_fingers(lm_list)
            print(total_fingers)

            if 0 < total_fingers <= len(overlay_images):
                overlay_image = overlay_images[total_fingers]
                h, w, _ = overlay_image.shape
                img[0:h, 0:w] = overlay_image 

            cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(total_fingers), (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        cv.waitKey(1)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()