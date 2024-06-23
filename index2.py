import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Load the image
image_path = '/Users/cpr/Desktop/01.jpg'  # Replace with your image path
image = io.imread(image_path)
image = image / 255.0  # Normalize the pixel values

# Function to perform SVD and reconstruct the image with k singular values
def svd_compress(image_channel, k):
    U, S, V = np.linalg.svd(image_channel, full_matrices=False)
    S = np.diag(S)
    compressed_image_channel = np.dot(U[:, :k], np.dot(S[:k, :k], V[:k, :]))
    return compressed_image_channel

# Compress each channel
k = 50 # Number of singular values to keep
compressed_image = np.zeros_like(image)
for i in range(3):  # Loop over the three channels: R, G, B
    compressed_image[:, :, i] = svd_compress(image[:, :, i], k)

# Clip the values to be in the valid range and denormalize
compressed_image = np.clip(compressed_image, 0, 1)

# Display the original and compressed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title(f'Compressed Image with k={k}')
plt.axis('off')
plt.imshow(compressed_image)

plt.show()
