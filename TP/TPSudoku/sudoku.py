import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

## résolution d'une grille de sudoku avec reconnaissance de caractères modele OCR

input_size = 48
model = load_model("model-OCR.h5")
classes = np.arange(0, 10)

# Chargement de l'image
def charger_image():
    # Récuperation de l'image
    img = cv.imread("sudoku1.jpg")
    return img

def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBif = cv.bilateralFilter(imgGray, 13, 20, 20)
    imgCanny = cv.Canny(imgBif, 30, 180) 
    return imgCanny

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
    dest_points = np.float32([[0, 900], [0, 0], [900, 0], [900, 900]])

    # Calculer la matrice de transformation de perspective
    perspective_matrix = cv.getPerspectiveTransform(np.float32(corner_points), dest_points)

    # Appliquer la transformation de perspective
    output_image = cv.warpPerspective(image, perspective_matrix, (900, 900))

    # Transposer et retourner l'image pour la rotation de -90 degrés
    output_image = cv.flip(output_image, 0)

    return output_image
            
# Split the image into 81 different images
def split_boxes(img):
    # output image en niveau de gris
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = np.vsplit(image, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv.resize(box, (input_size, input_size))/255.0
            cv.imshow("Box", box)
            cv.waitKey(0)
            boxes.append(box)
    print(len(boxes))
    return boxes

# Affichage de la prediction dans la console
def predict(boxes):
    # Boxes en niveau de gris
    boxes = np.array(boxes).reshape(-1, input_size, input_size, 1)
    print(len(boxes))
    # Prediction
    prediction = model.predict(boxes)
    print(prediction)
    # Classes
    predicted_numbers = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)
    print(predicted_numbers)
    # Reshape
    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
    print(board_num)
    
    return board_num

# Algorithme de résolution du sudoku (Backtracking)
def caseSuivante(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def estPossible(board, num, pos):
    # Vérification de la ligne
    for i in range(9):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    # Vérification de la colonne
    for i in range(9):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    # Vérification du carré 3x3
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False
    return True

def solve(board):
    find = caseSuivante(board)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1, 10):
        if estPossible(board, i, (row, col)):
            board[row][col] = i
            if solve(board):
                return True
            board[row][col] = 0
    return False

def displayNumbers(img, numbers, color=(0,255,0)):
    w = int(img.shape[1]/9)
    h = int(img.shape[0]/9)
    for i in range(9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                print(str(numbers[(j*9)+i]))
                cv.putText(img, str(numbers[(j*9)+i]), (i*w+int(w/2)-int((w/4)), int((j+0.7)*h)), 
                           cv.FONT_HERSHEY_COMPLEX, 2, color, 2, cv.LINE_AA)
    return img

# Affichage de l'image de base
def afficher_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)
    
# Affichage de l'image traitée
def afficher_image_traitee(img):
    cv.imshow("Image Traitee", img)
    cv.waitKey(0)
    
# Affichage de l'image avec les contours
def afficher_image_contours(img):
    cv.imshow("Image Contour", img)
    cv.waitKey(0)
    
# Affichage des splits
def afficher_splits(boxes):
    for box in boxes:
        cv.imshow("Box", box)
        
def affichage_solution_sudoku(img, numbers):
    img_with_numbers = displayNumbers(img, numbers)
    cv.imshow("Image avec les chiffres", img_with_numbers)
    cv.waitKey(0)

# Main
def main():
    # initialisation des images
    img = charger_image()
    img2 = preProcessing(img)
    
    # Trouver le plus grand contour
    max_contour = find_largest_contour(img2)
    
    # Trouver les coins du contour
    corner_points = find_corner_points(max_contour)
    
    # Appliquer la transformation de perspective
    output_image = apply_perspective_transform(img, corner_points)
    
     # Prédiction du Sudoku
    board_num = predict(split_boxes(output_image))
    
    # Résolution du Sudoku
    solved_board = solve(board_num)
    
    # Affichage des images et de la solution
    afficher_image(img)
    afficher_image_traitee(img2)
    afficher_image_contours(output_image)
    affichage_solution_sudoku(output_image, solved_board)
    
    
if __name__ == "__main__":
    main()    
