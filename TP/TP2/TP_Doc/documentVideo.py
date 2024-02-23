import cv2 as cv
import numpy as np

def preprocess_image_video(frame):
    # Convertir en niveau de gris
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien
    img_blur = cv.GaussianBlur(gray, (1, 1), 0)
    
    # Dilatation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(img_blur, kernel, iterations=1)
    
    return gray, img_blur, dilate

def find_largest_contour_video(frame):
    # Appliquer un seuil à l'image dilatée
    ret, thresh = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
    
    # Trouver les contours dans l'image binaire
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Trouver le plus grand contour
    max_contour = max(contours, key=cv.contourArea)
    
    return max_contour

def find_corner_points_video(contour):
    
    # Approximation du contour pour obtenir les coins
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    
    return approx

def apply_perspective_transform_video(frame, corner_points):
    
    # if corner_points[1][0] <  corner_points[2][0]:
    #     corner_points[[1, 2]] = corner_points[[2, 1]]
        
    width, height = 300, 300

    corner_points = np.float32(corner_points)   
    dest_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # dest_points = np.array(dest_points, dtype=np.float32)

    # Calculer la matrice de transformation de perspective
    perspective_matrix = cv.getPerspectiveTransform(corner_points, dest_points)

    # Appliquer la transformation de perspective
    output_frame = cv.warpPerspective(frame, perspective_matrix, (300, 300))

    return output_frame

def main():
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        # Prétraiter l'image
        gray, img_blur, dilate = preprocess_image_video(frame)

        # Trouver le plus grand contour
        max_contour = find_largest_contour_video(dilate)
        if max_contour is None:
            output_frame = frame.copy()
        else:
            # Trouver les coins
            corner_points = find_corner_points_video(max_contour)
            
            if len(corner_points) != 4:
                output_frame = frame.copy()
            else:
                # Appliquer la transformation de perspective
                output_frame = apply_perspective_transform_video(frame, corner_points)

            # Afficher les images
            cv.imshow("frame", frame)
            # cv.imshow("gray", gray)
            # cv.imshow("img_blur", img_blur)
            # cv.imshow("dilate", dilate)
        cv.imshow("output_frame", output_frame)

        # Quitter le programme
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()