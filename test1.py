import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# load image
img_path = r".\image_tests\output3.jpg"
img = cv2.imread(img_path, 1)

# height, width, depth and ratio
h, w, d = img.shape
resized_w = 1400
ratio = resized_w / w

# resize image
resized = imutils.resize(img, width=resized_w)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# blurr the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Identify circles
all_circles = cv2.HoughCircles(
    blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=0.1,
    minDist=10,
    param1=20,
    param2=10,
    minRadius=11,
    maxRadius=12,
)
circles = np.uint16(np.around(all_circles))
print("It found " + str(circles.shape[1]) + " circles on the pi&d")

print(all_circles)

count = 0
img_circles = resized.copy()
for circle in circles[0]:
    # Annotate circle and centroid
    cv2.circle(img_circles, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    cv2.circle(img_circles, (circle[0], circle[1]), 2, (255, 0, 0), -2)

    # Annotate text
    offset_txt = int(circle[2] * 1.5)
    cv2.putText(
        img_circles,
        "Circle " + str(count),
        (circle[0] - offset_txt, circle[1] + offset_txt),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 0, 0),
        1,
    )
    count += 1

cv2.imshow("Image", img_circles)
cv2.waitKey(0)