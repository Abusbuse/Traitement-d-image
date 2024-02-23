import cv2 as cv
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from handDetector import HandDetector

def get_audio_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

def set_volume(volume_interface, distance):
    normalized_distance = min(distance / 100, 1.0)
    volume_interface.SetMasterVolumeLevelScalar(normalized_distance, None)

def draw_line(image, lm_list):
    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    thumb_x, thumb_y = thumb_tip[1], thumb_tip[2]
    index_x, index_y = index_tip[1], index_tip[2]
    cv.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

def draw_circle_center(image, lm_list, distance, distance_threshold=40):
    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    thumb_x, thumb_y = thumb_tip[1], thumb_tip[2]
    index_x, index_y = index_tip[1], index_tip[2]
    center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
    if distance < distance_threshold:
        cv.circle(image, (center_x, center_y), 10, (0, 255, 0), cv.FILLED)
    else:
        cv.circle(image, (center_x, center_y), 10, (0, 0, 0), cv.FILLED)

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()
    volume_interface = get_audio_interface()

    # Variable pour activer ou désactiver l'affichage supplémentaire
    affichage_supplementaire = True

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        
        if len(lm_list) != 0:
            tTip = lm_list[4]
            iTip = lm_list[8]
            t_x, t_y = tTip[1], tTip[2]
            i_x, i_y = iTip[1], iTip[2]
            distance = ((t_x - i_x) ** 2 + (t_y - i_y) ** 2) ** 0.5
            print("Distance entre les doigts", distance)
            
            if affichage_supplementaire:
                draw_line(img, lm_list)
                draw_circle_center(img, lm_list, distance)
                
            set_volume(volume_interface, distance)
            
            # Calculer le pourcentage du volume actuel
            current_volume = volume_interface.GetMasterVolumeLevelScalar() * 100

            # Dessiner la barre de volume
            bar_length = 400
            bar_height = 20
            bar_x = (img.shape[1] - bar_length) // 2
            bar_y = img.shape[0] - 50
            cv.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (0, 0, 0), cv.FILLED)
            filled_length = int(bar_length * (current_volume / 100))
            cv.rectangle(img, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), (0, 255, 0), cv.FILLED)

            # Afficher le pourcentage du volume
            text_size = cv.getTextSize(f"Volume: {int(current_volume)}%", cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = bar_y - 10
            cv.putText(img, f"Volume: {int(current_volume)}%", (text_x, text_y), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Activer ou désactiver l'affichage supplémentaire en appuyant sur la touche 's'
            affichage_supplementaire = not affichage_supplementaire

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
