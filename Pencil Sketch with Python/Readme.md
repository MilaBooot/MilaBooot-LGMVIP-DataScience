# MilaBooot-LGMVIP-DataScience
# Let’s import it to get started with the task
import cv2
import os

# Now the next thing to do is to read the image
image = cv2.imread("jami.jpg")
![jami](https://user-images.githubusercontent.com/57688047/129746695-769890cc-48af-4551-832f-62672b11fc15.jpg)

# Now after reading the image, we will create a new image by converting the original image to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
![gray image](https://user-images.githubusercontent.com/57688047/129746779-12b7092f-a1cd-415b-9647-0aff4a6563a9.jpg)

# Now the next step is to invert the new grayscale image
inverted = 255 - gray_image
![inverted image](https://user-images.githubusercontent.com/57688047/129746823-c0960c2f-af74-4f36-ad29-e7620c233384.jpg)

# Now the next step in the process is to blur the image by using the Gaussian Function in OpenCV
blury_image = cv2.GaussianBlur(inverted, (23, 23), 0)

# Then the final step is to invert the blurred image, then we can easily convert the image into a pencil sketch:
blury_inverted = 255 - blury_image

pencil_sketech = cv2.divide(gray_image, blury_inverted, scale = 250)
![saved image](https://user-images.githubusercontent.com/57688047/129746945-76997db0-c72e-4289-a000-37ece912e214.jpg)

cv2.imshow("blury_image", blury_image)
#cv2.imshow("inverted", inverted)

#cv2.imshow("gray", gray_image)

#cv2.imshow("jami", image)

#cv2.imshow("pencil_sketech", pencil_sketech)

# Save images in JPG format
cv2.imwrite("saved image.jpg", pencil_sketech)
cv2.waitKey(0)
