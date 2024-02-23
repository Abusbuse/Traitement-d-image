import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    
    # Carré au milieu de l'image
    cv.rectangle(img,(200,200),(300,300),(255,255,255),3)
    
    # Cercle vide au milieu du carré
    cv.circle(img,(250,250), 50, (255,255,255), 3)
    
    # Triangle vide sur chaque coin du carré
    pts = np.array([[200,200],[300,200],[250,100]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(255,255,255))
    
    pts = np.array([[200,300],[300,300],[250,400]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(255,255,255))
    
    pts = np.array([[200,200],[200,300],[100,250]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(255,255,255))
    
    pts = np.array([[300,200],[300,300],[400,250]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(255,255,255))
    
    # Afficher l'image
    cv.imshow("Logo de fou furieux", img)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("logoOpenCV.png", img)
    cv.destroyAllWindows()
    
    # On sort quand q est cliqué
    if k == ord("q") or k == ord("Q"):
        sys.exit("Fin du programme")