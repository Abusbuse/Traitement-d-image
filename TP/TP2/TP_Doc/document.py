import cv2 as cv
import numpy as np

def preprocess_image(img):
    # Convertir en niveau de gris
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien
    img_blur = cv.GaussianBlur(gray, (1, 1), 0)
    
    # Dilatation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(img_blur, kernel, iterations=1)
    
    return gray, img_blur, dilate

def find_largest_contour(img):
    # Appliquer un seuil à l'image dilatée
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    
    # Trouver les contours dans l'image binaire
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Trouver le plus grand contour
    max_contour = max(contours, key=cv.contourArea)
    
    return max_contour

def find_corner_points(contour):
    # Approximation du contour pour obtenir les coins
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    
    # Récupérer les points des coins
    corner_points = [tuple(point[0]) for point in approx]
    
    return corner_points

def apply_perspective_transform(image, corner_points):
    # Définir les points de destination pour la transformation de perspective
    dest_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

    # Calculer la matrice de transformation de perspective
    perspective_matrix = cv.getPerspectiveTransform(np.float32(corner_points), dest_points)

    # Appliquer la transformation de perspective
    output_image = cv.warpPerspective(image, perspective_matrix, (300, 300))

    # Transposer et retourner l'image pour la rotation de -90 degrés
    output_image = cv.flip(output_image, 0)

    return output_image



def main():
    img = cv.imread("feuilleSurFondNoir.jpg")
    img = cv.resize(img, (680, 480))

    # Prétraiter l'image
    gray, img_blur, dilate = preprocess_image(img)

    # Trouver le plus grand contour
    max_contour = find_largest_contour(dilate)

    # Trouver les coins du contour
    corner_points = find_corner_points(max_contour)

    # Appliquer la transformation de perspective
    output_image = apply_perspective_transform(img, corner_points)

    # Afficher les fenêtres
    cv.namedWindow("Image")
    cv.imshow("Image", img)

    cv.namedWindow("Image Blur")
    cv.imshow("Image Blur", img_blur)

    cv.namedWindow("Image Gray")
    cv.imshow("Image Gray", gray)

    cv.namedWindow("Image Dilate")
    cv.imshow("Image Dilate", dilate)

    cv.namedWindow("Image Output")
    cv.imshow("Image Output", output_image)    

    # Attendre la pression de la touche 'esc' pour quitter
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
