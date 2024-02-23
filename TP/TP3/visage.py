import cv2 

def detect_face():
    # Chargez l'image
    img = cv2.imread("visages.jpg")

    # Convertissez l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chargez le classificateur Haar
    haar_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

    # Détectez les visages
    faces_rect = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.3, minNeighbors=10
    )

    # Affichez un rectangle autour de chaque visage
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Afficher l'image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    detect_face()