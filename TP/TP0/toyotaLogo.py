import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
    # Create a white image
    img = np.zeros((512,512,3), np.uint8)
    
    # Création du logo Toyota
    # Carré blanc dans le fond
    cv.rectangle(img,(0,0),(600,600),(255,255,255),-1)
    
    # Ovale noirs
    cv.ellipse(img,(256,255),(180,111),0,0,360,(0,0,0),2)
    cv.ellipse(img,(256,256),(108,34),90,0,360,(0,0,0),2)
    cv.ellipse(img,(256,256),(215,140),0,0,360,(0,0,0),2)
    cv.ellipse(img,(256,187),(137,65),0,0,360,(0,0,0),2)
    cv.ellipse(img,(256,256),(134,71),90,0,360,(0,0,0),2)
    cv.ellipse(img,(256,185),(113,39),0,0,360,(0,0,0),2)
    
    # Ovale principal blanc
    cv.ellipse(img,(256,256),(200,125),0,0,360,(255,255,255),20)
    # Ovale principal gris
    cv.ellipse(img,(256,256),(190,120),0,0,360,(125,125,125),15)
    
    # Ovale intérieur gris rotation 
    cv.ellipse(img,(256,256),(113,40),90,0,360,(125,125,125),7)
    # Ovale blanc
    cv.ellipse(img,(256,185),(125,55),0,0,360,(255,255,255),20)
    # Ovale gris
    cv.ellipse(img,(256,185),(118,44),0,0,360,(125,125,125),5)
    
    # Ovale intérieur blanc rotation 90°
    cv.ellipse(img,(256,256),(122,55),90,0,360,(255,255,255),25)
    
    # Lettre T en rouge 
    cv.putText(img,'TOYOTA',(1,510), cv.FONT_HERSHEY_SIMPLEX, 4.5,(0,0,255),10,cv.LINE_AA)    
    
    # Afficher l'image
    cv.imshow("Logo de fou furieux", img)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("logoOpenCV.png", img)
    cv.destroyAllWindows()
    
    # On sort quand q est cliqué
    if k == ord("q") or k == ord("Q"):
        sys.exit("Fin du programme")