import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('C:/Users/manci/OneDrive/Escritorio/images.jpg', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.medianBlur(image, 5)


circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.12, minDist=30,
                           param1=20, param2=40, minRadius=20, maxRadius=38)

output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.title('Círculos Detectados')
plt.show()
