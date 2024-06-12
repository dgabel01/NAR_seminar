import cv2
import numpy as np

# Učitaj sliku
image = cv2.imread('Togir4.jpg')


if image is None:
    print("Error: Could not read the image.")
    exit()

# Primjeni 7x7
kernel_size = (7, 7)
mean_filter = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
filtered_image = cv2.filter2D(image, -1, mean_filter)

# Pohrani obrađenu sliku
cv2.imwrite('filtered_image.jpg', filtered_image)

# Prikaži originalnu i promijenjenu sliku
cv2.imshow('Originalna slika', image)
cv2.imshow('Filtrirana slika', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
