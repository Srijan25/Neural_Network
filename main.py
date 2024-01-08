import cv2
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set(color_codes=True)# Read the image
image = cv2.imread('Flower.jpg') #--imread() helps in loading an image into jupyter including its pixel valuesplt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()
image.shape
image[0][0]
# Convert image to grayscale. The second argument in the following step is cv2.COLOR_BGR2GRAY, which converts colour image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Original Image")
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()
gray.shape
import numpy as np
data = np.array(gray)
flattened = data.flatten()
flattened.shape
flattened