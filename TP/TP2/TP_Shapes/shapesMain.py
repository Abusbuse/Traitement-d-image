import cv2 as cv
import numpy as np

def loadImage():
    # Récupération de l'image
    img = cv.imread(cv.samples.findFile('shapes.png'))
    
    return img

def main(img):
    while True:
        # Image original
        cv.namedWindow("Original")

        # Convertir l'image en niveau de gris
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.namedWindow("Gray")

        # Lisser l'image grey
        grayBlur = cv.GaussianBlur(gray, (5, 5), 0)
        cv.namedWindow("Gray Blur")

        # Détection des contours
        edges = cv.Canny(img, 50, 150)
        
        # Contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_image = img.copy()
        
        # Dessiner les contours sur une copie de l'image originale
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv.namedWindow("Contours")
        
        # Affichage d'une boîte englobante pour chaque forme géométrique
        imgBoite = img.copy()
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(imgBoite, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv.namedWindow("Contours et boites")

       # Ajout de texte indiquant la forme géométrique
        imgBoiteTexte = img.copy()
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(imgBoiteTexte, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculer une approximation polygonale
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)

            # Utiliser le nombre de sommets pour étiqueter les formes
            shape_label = ""
            if len(approx) == 3:
                shape_label = "Triangle"
            elif len(approx) == 4:
                # Vérifier si c'est un carré en vérifiant le rapport hauteur-largeur
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_label = "Carre"
                else:
                    shape_label = "Rectangle"
            elif len(approx) >= 7:
                shape_label = "Cercle"

            # Dessiner la boîte englobante
            cv.rectangle(imgBoiteTexte, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Ajouter le texte avec l'étiquette de la forme
            cv.putText(imgBoiteTexte, shape_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv.namedWindow("Boite et texte")

        # Affichage de l'image dans une fenêtre
        cv.imshow("Original", img)
        cv.imshow("Gray", gray)
        cv.imshow("Gray Blur", grayBlur)
        cv.imshow("Contours", contour_image)
        cv.imshow("Contours et boites", imgBoite)
        cv.imshow("Boite et texte", imgBoiteTexte)
        
        if cv.waitKey(1) == ord('q'):
            break
        
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(loadImage())
