import numpy as np
import cv2
import math
from scipy import ndimage

img_before = cv2.imread('dataset/resized/rakib_image_2.jpg')

cv2.imshow("Before", img_before)    
key = cv2.waitKey(0)

img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []

for [[x1, y1, x2, y2]] in lines:
    # cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

# cv2.imshow("Detected lines", img_before)    
# key = cv2.waitKey(0)

median_angle = np.median(angles)
# img_rotated = ndimage.rotate(img_before, median_angle)

print(f"Angle is {median_angle:.04f}")

# img = cv2.imread('messi5.jpg',0)
rows,cols, channels = img_before.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),median_angle,1)
dst = cv2.warpAffine(img_before,M,(cols,rows))

cv2.imwrite('dataset/resized/aarrakib_image_1.jpg', dst)  