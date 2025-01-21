import cv2
import os
from ultralytics import YOLO

# Modell laden
model = YOLO("radcog-0.1.4.pt")  # Du kannst auch yolov5m.pt, yolov5l.pt oder yolov5x.pt ausprobieren

# Load all files from input dir
for image_path in os.listdir("input"):

    image_path = "input/" + image_path
    image = cv2.imread(image_path)

    # Objekterkennung durchführen
    results = model(image_path, conf=0.05)



    # Ergebnisse verarbeiten und anzeigen
    # for result in results:
    #for box in results[0].boxes:
    if(len(results[0].boxes) > 0):
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0]  # Koordinaten der Box
        conf = box.conf[0]           # Vertrauen
        cls = box.cls[0]             # Klassen-ID

        print(f"Gefunden: {label} mit {conf:.3f} Konfidenz")

        # Zeichne die Box auf das Bild
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.3f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Bild anzeigen
    cv2.imshow("Ergebnisse", image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
