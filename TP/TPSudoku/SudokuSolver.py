# Author: Flavien Marck (@flaw_9)

import cv2 as cv
import numpy as np
import os
from tensorflow.keras.models import load_model
from algoSudoku import AlgoSudoku

WIDTH, HEIGHT = 900, 900

input_size = 48
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def filtre_canny(img):
    # On floute l'image
    img_blur = cv.GaussianBlur(img, (5, 5), 1)
    # On détecte les contours
    img_canny = cv.Canny(img_blur, 50, 50)
    return img_canny

def reshuffle(points):
    # On calcule la somme des coordonnées de chaque point
    # On renvoie le tableau dans l'ordre croissant

    # On calcule la somme des coordonnées de chaque point
    sums = []
    for index, point in enumerate(points):
        sums.append([point[0][0] + point[0][1], index])

    # On trie le tableau
    sums.sort()

    new_points = np.zeros((len(points), 2), dtype=np.int32)
    for i in range(len(sums)):
        new_points[i] = points[sums[i][1]]

    # Si le point 1 est plus à droite que le point 2
    # On les échange
    if new_points[1][0] < new_points[2][0]:
        new_points[[1, 2]] = new_points[[2, 1]]

    return new_points

def detect_contours(img_canny):
    # On récupère les contours
    contours, _ = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    points = []

    biggest = None
    biggest_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest = contour
    
    if biggest is not None:
        # On trouve les points à partir du contour trouvé, grâce à PolyDP
        points = cv.approxPolyDP(
            biggest, 0.02 * cv.arcLength(biggest, True), True)
    
    return points

def detect_grille(img_canny, img) -> (np.ndarray, np.ndarray, bool):
    found = False
    canny, origin, result = img_canny.copy(), img.copy(), img.copy()
    # On récupère les contours
    points = detect_contours(canny)    
    # We make sure it forms something close enough to a rectangle
    if len(points) == 4 and cv.isContourConvex(points):
        # On dessine les points
        for point in points:
            cv.circle(origin, (point[0][0], point[0]
                        [1]), 10, (0, 255, 0), cv.FILLED)
            
        # On calcule les dimensions de l'image résultat
        # À partir des 4 points trouvés
        points = reshuffle(points)

        # On transforme les points en tableau de float32

        # La longueur = points[0] -> points[1]
        # La largeur = points[0] -> points[2]
        w = int(np.sqrt((points[0][0] - points[1][0])
                ** 2 + (points[0][1] - points[1][1]) ** 2))
        h = int(np.sqrt((points[0][0] - points[2][0])
                ** 2 + (points[0][1] - points[2][1]) ** 2))

        points = np.float32(points)

        perspective = cv.getPerspectiveTransform(
            points, np.float32([[0, 0], [w, 0], [0, h], [w, h]]))
        result = cv.warpPerspective(img, perspective, (w, h))
        result = cv.resize(result, (WIDTH, HEIGHT))
        found = True

    return origin, result, found

def split_into_cells(img: np.ndarray):
    # On découpe l'image en 81 cases
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cells = []
    rows = np.vsplit(img, 9)
    for row in rows:
        cols = np.hsplit(row, 9)
        for cell in cols:
            cell = cv.resize(cell, (input_size, input_size)) / 255.0
            cells.append(cell)
    return cells

def get_cells_values(cells: list, model) -> list:
    prediction = model.predict(cells)
    pred = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = classes[index]
        pred.append(predicted_number)
    return pred

def display_result_on_grid(solver: AlgoSudoku, final_result: np.ndarray):
    # On créé une image noire sur laquelle on écrit les valeurs
    img = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
    img.fill(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            if solver.grid[i][j] != 0 and solver.original_grid[i][j] == 0:
                cv.putText(img, str(solver.grid[i][j]), (j * 100 + 30, i * 100 + 70),
                           font, 2, (0, 255, 0), 3, cv.LINE_AA)
    
    return img

def result_to_perspective(img: np.ndarray, result: np.ndarray):
    # On récupère les contours de la grille sur l'image de départ
    points = detect_contours(img)
    points = reshuffle(points)

    w, h = result.shape[1], result.shape[0]
    
    points = np.float32(points)
    perspective = cv.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [0, h], [w, h]]), points)
    
    # On créé une nouvelle image noire, et on y ajoute le résultat avec la perspective
    black = cv.warpPerspective(result, perspective, (img.shape[1], img.shape[0]))
    return black

def main():
    # Lecture de l'image    
    filename = "sudoku1.jpg"
    img = cv.imread(os.path.join(os.path.dirname(__file__), filename), 1)

    img_result = img.copy()
    img_grille = img.copy()

    img_canny = filtre_canny(img)

    (img_grille, grille, found) = detect_grille(img_canny, img_grille)

    if not found:
        print("Grille non trouvée")
        return
    
    # Now, we'll split the image into 81 cells
    cells = split_into_cells(grille)
    cells = np.array(cells).reshape(-1, input_size, input_size, 1)

    model = load_model(os.path.join(os.path.dirname(__file__), "model-OCR.h5"))
    values = get_cells_values(cells, model)

    # On fait un tableau de 9 par 9 au format uint8
    values = np.array(values).reshape(9, 9)

    solver = AlgoSudoku(np.array(values).copy())
    if not solver.solve():
        print("Pas de solution")
        return
    
    print(solver.original_grid)
    print(solver.grid)

    grille_result = cv.bitwise_or(grille, display_result_on_grid(solver, img_result))

    perspect_result = result_to_perspective(img_canny, display_result_on_grid(solver, img_result))

    # Bitwise or image de base et perspective
    img_result = cv.bitwise_or(img_result, perspect_result)

    # --
    # -- AFFICHAGE --
    # --
    cv.imshow("Grille", img_grille)
    cv.imshow("Grille result", grille_result)
    cv.imshow("Résultat", img_result)
    cv.imshow("Perspective", perspect_result)

    while 1:

        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()