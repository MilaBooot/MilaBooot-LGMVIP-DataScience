# Let’s import it to get started with the task
import cv2
import os

# Now the next thing to do is to read the image
image = cv2.imread("jami.jpg")

# Now after reading the image, we will create a new image by converting the original image to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Now the next step is to invert the new grayscale image
inverted = 255 - gray_image

# Now the next step in the process is to blur the image by using the Gaussian Function in OpenCV
blury_image = cv2.GaussianBlur(inverted, (23, 23), 0)

# Then the final step is to invert the blurred image, then we can easily convert the image into a pencil sketch:
blury_inverted = 255 - blury_image
pencil_sketech = cv2.divide(gray_image, blury_inverted, scale = 250)

cv2.imshow("blury_image", blury_image)
#cv2.imshow("inverted", inverted)
#cv2.imshow("gray", gray_image)
#cv2.imshow("jami", image)
#cv2.imshow("pencil_sketech", pencil_sketech)
# Save images in jpg format
cv2.imwrite("inverted image.jpg", inverted)
cv2.imwrite("blury_inverted.jpg", blury_inverted)
cv2.imwrite("gray image.jpg", gray_image)
cv2.imwrite("jami image.jpg", image)

cv2.imwrite("saved image.jpg", pencil_sketech)
cv2.waitKey(0)