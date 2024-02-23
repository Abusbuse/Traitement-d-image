import cv2 as cv
import mediapipe as mp
import time
from handDetector import HandDetector
import mouse
    
# fonction qui detecte si l'index et le major sont levés
def index_major_up(img, lm_list):
    if len(lm_list) != 0:
        # si l'index est levé
        if lm_list[8][2] < lm_list[6][2]:
            cv.circle(img, (lm_list[8][1], lm_list[8][2]), 15, (255, 0, 0), cv.FILLED)
            # si le major est levé
            if lm_list[12][2] < lm_list[10][2]:
                cv.circle(img, (lm_list[12][1], lm_list[12][2]), 15, (0, 255, 0), cv.FILLED)
    return img

# Dessine un rectangle dans la fenetre qui va etre une zone de clic 
def draw_click_zone(img):
    h, w, c = img.shape
    # Rectangle vide au centre de l'image 960x540
    cv.rectangle(img, (320, 180), (960, 540), (100, 100, 100), 2)
    return img

# Header 1280x 125 divisé en 4 parties, trois pour les couleurs rouge bleu et vert et une gomme qui efface tout
def header(img):
    h, w, c = img.shape
    cv.rectangle(img, (0, 0), (w, 125), (0, 0, 0), cv.FILLED)
    cv.putText(img, "Rouge", (150, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.putText(img, "Vert", (450, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.putText(img, "Bleu", (750, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv.putText(img, "Effacer", (1050, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    return img

def dessinerAvecDoigts(img, lm_list):
    # Si l'index et le major sont levés
    if len(lm_list) != 0:
        if lm_list[8][2] < lm_list[6][2] and lm_list[12][2] < lm_list[10][2]:
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[12][1], lm_list[12][2]
            # Si le clic est dans la zone de dessin
            if 320 < x1 < 960 and 180 < y1 < 540 and 320 < x2 < 960 and 180 < y2 < 540:
                cv.circle(img, (x1, y1), 15, (255, 0, 0), cv.FILLED)
                cv.circle(img, (x2, y2), 15, (0, 255, 0), cv.FILLED)
                # Si le clic est dans la zone rouge
                if 0 < x1 < 320:
                    cv.circle(img, (x1, y1), 15, (255, 0, 0), cv.FILLED)
                    mouse.click(x1, y1, button=2)

                # Si le clic est dans la zone verte
                if 320 < x1 < 640:
                    cv.circle(img, (x1, y1), 15, (0, 255, 0), cv.FILLED)
                    mouse.click(x1, y1, button=2)

                # Si le clic est dans la zone bleue
                if 640 < x1 < 960:
                    cv.circle(img, (x1, y1), 15, (0, 0, 255), cv.FILLED)
                    mouse.click(x1, y1, button=2)

                # Si le clic est dans la zone effacer
                if 960 < x1 < 1280:
                    cv.circle(img, (x1, y1), 15, (255, 255, 255), cv.FILLED)
                    mouse.click(x1, y1)            
    # Si l'index est levé alors on dessine
    if len(lm_list) != 0:
        if lm_list[8][2] < lm_list[6][2]:
            x1, y1 = lm_list[8][1], lm_list[8][2]
            # Si le clic est dans la zone de dessin
            if 320 < x1 < 960 and 180 < y1 < 540:
                cv.circle(img, (x1, y1), 15, (255, 0, 0), cv.FILLED)
                mouse.move(x1, y1)
    return img


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    # Rezi de la fenetre 1280x720
    cap.set(3, 1280)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        
        # Affiche la position du landmark 4
        if len(lm_list) != 0:
            print(lm_list[4])

        img = header(img)
        
        img = draw_click_zone(img)
        
        img = index_major_up(img, lm_list)
        
        img = dessinerAvecDoigts(img, lm_list)
           
        # Calcul du FPS 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Affiche les FPS        
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows

if __name__ == "__main__":
    main()