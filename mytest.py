import skimage.draw
import numpy as np
import cv2

img = np.zeros((128, 128, 1))
img = np.uint8(img)

rr, cc, value = skimage.draw.line_aa(20,20,0,100)
print(rr, cc, value)
img[rr, cc, 0] = 255

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()