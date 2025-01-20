import cv2
import numpy as np

# Bild laden
image = cv2.imread("ball.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Farbgrenzen definieren (z. B. Weiß/Grau und Streifenfarben)
# Passen Sie diese Werte basierend auf Ihrer Farbwertanalyse an
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 80, 255])

lower_color = np.array([0,50,50])  # Beispiel: Orange
upper_color = np.array([10,255,255])  # Beispiel: Orange

# Masken erstellen
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Kombinierte Maske
combined_mask = cv2.bitwise_or(white_mask, color_mask)

# Maske auf Bild anwenden
filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

cv2.imwrite("ballf.png", filtered_image)

# Ergebnisse anzeigen
cv2.imshow("Originalbild", image)
cv2.imshow("Gefiltertes Bild", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
