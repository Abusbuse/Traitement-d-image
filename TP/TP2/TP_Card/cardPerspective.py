import cv2 as cv
import numpy as np

def main():
    # Récuperation de l'image
    img = cv.imread("cards.jpg")
    imgB = img.copy()
    imgC = img.copy()
    
    points = np.int32([[111, 219], [287, 188], [154, 482], [352, 440]])
    
    while True:
        
        for point in points:
            cv.circle(img, tuple(point), 5, (0, 255, 0), -1)
            cv.putText(img, str(point), tuple(point), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
        # ImageB
        for point in points:
            cv.circle(imgB, tuple(point), 5, (255, 0, 0), -1)
            
        perspective = cv.getPerspectiveTransform(np.float32(points), np.float32([[0, 0], [176, 0], [0, 250], [176, 250]]))
        
        goOutput = cv.warpPerspective(imgC, perspective, (176, 250))
            
        # Affichage des fenêtres
        cv.namedWindow("ImageA")
        cv.imshow("ImageA", img)
        
        cv.namedWindow("ImageB")
        cv.imshow("ImageB", imgB)
        
        cv.namedWindow("Output")
        cv.imshow("Output", goOutput)
        # q pour quitter le programme
        if cv.waitKey(1) == ord('q'):
            break
    
    # Fermeture des fenêtres
    img.release()
    cv.waitKey(1)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    