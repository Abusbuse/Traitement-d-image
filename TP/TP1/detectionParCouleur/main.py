import cv2 as cv

def loadImage():
    # On récupère l'image
    img = cv.imread(cv.samples.findFile('car.png'))
    
    return img

def convertToHSV(img):
    # On convertit l'image en HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Affichage de l'image dans une fenêtre
    cv.imshow("HSV", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return hsv

def maskHSV(img):
    # On convertit l'image en HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Ajout de trackbars pour modifier les valeurs HSV et donc changer la couleur de l'image
    cv.namedWindow("HSV")
    cv.createTrackbar("Hmin", "HSV", 0, 179, lambda x: x)
    cv.createTrackbar("Smin", "HSV", 0, 255, lambda x: x)
    cv.createTrackbar("Vmin", "HSV", 0, 255, lambda x: x)
    cv.createTrackbar("Hmax", "HSV", 179, 179, lambda x: x)
    cv.createTrackbar("Smax", "HSV", 255, 255, lambda x: x)
    cv.createTrackbar("Vmax", "HSV", 255, 255, lambda x: x)
    
    while True:
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
        res = cv.bitwise_and(img, img, mask=mask)
        
        # On affiche l'image modifiée
        cv.namedWindow("HSV")
        cv.imshow("HSV", mask)
        cv.namedWindow("Original")
        cv.imshow("Original", img)
        cv.namedWindow("Result")
        cv.imshow("Result", res)
        
        # On attend que l'utilisateur appuie sur la touche 'q' pour quitter
        if cv.waitKey(1) == ord('q'):
            break
    

    # Affichage de l'image dans une fenêtre
    cv.imshow("HSV", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return hsv

if __name__ == "__main__":
    
    maskHSV(loadImage())