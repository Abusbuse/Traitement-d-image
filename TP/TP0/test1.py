#Faire un logo tout simple avec openCV
import cv2 as cv 
import sys
import numpy as np


# img = cv.imread(cv.samples.findFile("belugaCatPDP.jpg"))

# if img is None:
#  sys.exit("Could not read the image.")


# # Resize window 300x300
# cv.namedWindow("Display window", cv.WINDOW_NORMAL)
# cv.resizeWindow("Display window", 300, 300)
# cv.imshow("Display window", img)
# k = cv.waitKey(0)


# if k == ord("s"):
#  cv.imwrite("starry_night.png", img)

# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(447,63), 63, (0,0,255), -1)
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

# Afficher l'image
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
 cv.imwrite("logoOpenCV.png", img)
cv.destroyAllWindows()