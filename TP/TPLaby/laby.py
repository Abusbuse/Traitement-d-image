import numpy as np
import cv2 as cv
from collections import deque

class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))

def mouseCallback(event, x, y, flags, param):
    global p, start, end
    if event == cv.EVENT_LBUTTONUP:
        if p == 0:
            cv.circle(imgRGB, (x, y), 5, (0, 0, 255), -1)
            start = Point(x, y)
            p += 1
        elif p == 1:
            cv.circle(imgRGB, (x, y), 5, (0, 255, 0), -1)
            end = Point(x, y)
            p += 1
        else:
            print("Trop de points cliqué")

def BFS(start, end, imgBinaire):  # Ajouter imgBinaire en paramètre
    queue = deque()
    parent = {}  # Utiliser un dictionnaire au lieu d'un ensemble
    queue.append(start)
    couleur = (0, 255, 255)  # Couleur pour le chemin trouvé
    
    while queue:
        current = queue.popleft()
        if current.x == end.x and current.y == end.y:
            print("Trouvé")
            chemin = []
            while not current.__eq__(start):
                chemin.append(current)
                current = parent[current]
            
            chemin.append(start)
            cv.circle(imgRGB, (start.x, start.y), 5, (255, 0, 0), -1)
            cv.circle(imgRGB, (end.x, end.y), 5, (255, 0, 0), -1)
            chemin_points = [(p.x, p.y) for p in chemin]
            cv.polylines(imgRGB, [np.array(chemin_points, np.int32)], False, (255, 0, 0), 1)
            break
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i == 0 or j == 0) and (i != 0 or j != 0):
                    next_point = Point(current.x + i, current.y + j)
                    if 0 <= next_point.x < imgBinaire.shape[1] and 0 <= next_point.y < imgBinaire.shape[0] and imgBinaire[next_point.y, next_point.x] == 255 and next_point not in parent:
                        parent[next_point] = current  # Enregistrer le parent de next_point
                        queue.append(next_point)
                        imgRGB[next_point.y, next_point.x] = couleur
                        imgBinaire[next_point.y, next_point.x] = 150

def main():
    global imgRGB, p, start, end
    p = 0
    img = cv.imread(cv.samples.findFile("laby3.png"))
    imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #image de base en gris
    (retVal, imgBinaire) = cv.threshold(imgGrey, 200, 255, cv.THRESH_BINARY)
    imgRGB = cv.cvtColor(imgBinaire, cv.COLOR_GRAY2BGR)

    cv.imshow("Color", imgRGB)
    cv.setMouseCallback("Color", mouseCallback)

    while True:
        if p >= 2:
            BFS(start, end, imgBinaire)  # Passer imgBinaire en paramètre
            p = -1  # Réinitialiser p pour éviter une boucle continue
        cv.imshow("Color", imgRGB)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
