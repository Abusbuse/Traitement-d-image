import cv2

def detect_license_plates():
    # Ouvrez la caméra
    capture = cv2.VideoCapture(0)

    # Chargez le classificateur Haar
    haar_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_licence_plate_rus_16stages.xml")

    # Boucle sur le flux vidéo
    while True:
        # Capturez une image
        success, img = capture.read()

        # Convertissez l'image en niveaux de gris
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détectez les plaques d'immatriculation
        plates_rect = haar_cascade.detectMultiScale(
            gray_img, scaleFactor=1.2, minNeighbors=5
        )

        # Affichez un rectangle autour de chaque plaque d'immatriculation
        for (x, y, w, h) in plates_rect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
        # Afficher l'image
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
                    

        # Attendez une touche
        key = cv2.waitKey(1)

        # Si la touche `q` est pressée, arrêtez le programme
        if key == ord("q"):
            break

    # Fermez la caméra
    capture.release()

if __name__ == "__main__":
    detect_license_plates()
