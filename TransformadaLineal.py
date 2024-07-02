import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('C:/Users/manci/OneDrive/Escritorio/Caja3.jpg', cv2.IMREAD_GRAYSCALE)

# Detectar bordes usando el detector de bordes de Canny
edges = cv2.Canny(image, 50, 200, apertureSize=3)

# Aplicar la Transformada de Hough
lines = cv2.HoughLines(edges, 1, np.pi / 100, 200)

# Crear una imagen a color para dibujar las líneas
line_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Dibujar las líneas detectadas
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar la imagen original y la imagen con líneas detectadas
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)), plt.title('Líneas Detectadas')
plt.show()