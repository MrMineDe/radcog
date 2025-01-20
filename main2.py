import cv2
from ultralytics import YOLO

# Modell laden (vortrainiertes YOLOv5-Modell)
model = YOLO("yolov8x.pt")  # Du kannst auch yolov5m.pt, yolov5l.pt oder yolov5x.pt ausprobieren

# Bild laden
image_path = "ball.png"  # Pfad zu deinem Bild
image = cv2.imread(image_path)


# Objekterkennung durchführen
results = model(image_path, conf=0.1)


# Ergebnisse verarbeiten und anzeigen
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Koordinaten der Box
        conf = box.conf[0]           # Vertrauen
        cls = box.cls[0]             # Klassen-ID

        # Klasse prüfen (z.B. "Ball")
        label = model.names[int(cls)]
        #if label == "sports ball":  # YOLO erkennt Bälle als "sports ball"
        print(f"Gefunden: {label} mit {conf:.2f} Konfidenz")

        # Zeichne die Box auf das Bild
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Bild anzeigen
cv2.imshow("Ergebnisse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
