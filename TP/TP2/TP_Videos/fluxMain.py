import cv2 as cv
import numpy as np

def main():
    # Récupération du flux vidéo
    cap = cv.VideoCapture(0)
    
    points = []
    
    # Ajout de trackbars pour modifier les valeurs HSV et donc changer la couleur de l'image
    cv.namedWindow("HSV")
    
    cv.createTrackbar("Hmin", "HSV", 0, 179, lambda x: x)
    cv.createTrackbar("Smin", "HSV", 0, 255, lambda x: x)
    cv.createTrackbar("Vmin", "HSV", 0, 255, lambda x: x)
    
    cv.createTrackbar("Hmax", "HSV", 179, 179, lambda x: x)
    cv.createTrackbar("Smax", "HSV", 255, 255, lambda x: x)
    cv.createTrackbar("Vmax", "HSV", 255, 255, lambda x: x)
    
    #Bleu
    # Régler les valeurs HSV
    cv.setTrackbarPos("Hmin", "HSV", 107)
    cv.setTrackbarPos("Smin", "HSV", 208)
    cv.setTrackbarPos("Vmin", "HSV", 114)
    
    cv.setTrackbarPos("Hmax", "HSV", 163)
    cv.setTrackbarPos("Smax", "HSV", 255)
    cv.setTrackbarPos("Vmax", "HSV", 255)
    
    while True:
        ret, frame = cap.read()
        # Flip de l'image
        frame = cv.flip(frame, 1)
        
        # Conversion du flux vidéo en HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # On récupère les valeurs des trackbars
        hmin = cv.getTrackbarPos("Hmin", "HSV")
        smin = cv.getTrackbarPos("Smin", "HSV")
        vmin = cv.getTrackbarPos("Vmin", "HSV")
        hmax = cv.getTrackbarPos("Hmax", "HSV")
        smax = cv.getTrackbarPos("Smax", "HSV")
        vmax = cv.getTrackbarPos("Vmax", "HSV")
        
        # Création d'un masque HSV
        mask = cv.inRange(hsv, (hmin, smin, vmin), (hmax, smax, vmax))
        
        # Application du mask sur l'image de base
        res = cv.bitwise_and(frame, frame, mask=mask)
        
        # Faire un lissage de l'image
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel)
        
        # Ajout d'un contour sur la partie bleue de l'image
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Affichage d'une boîte englobante pour chaque contour
        imgBoite = frame.copy()
        for contour in contours:
            # Ne prendre que les contours de taille suffisante
            if cv.contourArea(contour) < 1000:
                continue
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(imgBoite, (x, y), (x + w, y + h), (0, 255, 0), 2)
            points.append((x + w // 2, y + h // 2))
            
        # Récupération des points pour le centre de la boite englobante et ajout dans la liste
        for point in points:
            cv.circle(imgBoite, point, 5, (255, 0, 0), -1)
        
        cv.namedWindow("RES")
        cv.imshow("RES", res)
        
        # Resize de la fenêtre HSV
        cv.resizeWindow("HSV", 640, 480)
        cv.imshow("HSV", mask)
        
        cv.namedWindow("Result")
        cv.imshow("Result", imgBoite)
        
        # q pour quitter le programme
        if cv.waitKey(1) == ord('q'):
            break
        # r pour reset la liste des points
        elif cv.waitKey(1) == ord('r'):
            points = []
    
    # Fermeture des fenêtres
    cap.release()
    cv.waitKey(1)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
