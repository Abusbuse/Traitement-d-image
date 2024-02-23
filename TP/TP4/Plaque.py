import cv2

def detect_license_plates():
    # Chargez l'image
    img = cv2.imread("russian2.jpg")

    # Convertissez l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chargez le classificateur Haar
    haar_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_licence_plate_rus_16stages.xml")

    # Détectez les plaques d'immatriculation
    plates_rect = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=9
    )

    # Affichez un rectangle autour de chaque plaque d'immatriculation
    for (x, y, w, h) in plates_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Afficher l'image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    detect_license_plates()
