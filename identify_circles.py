import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import imutils
import re

# load image
img_path = r".\image_tests\output1.jpg"
img = cv2.imread(img_path, 1)

# height, width, depth and ratio
h, w, d = img.shape
# resized_w = 1400
# ratio = resized_w / w

resize_factor = 1

# resize image
resized = imutils.resize(img, width=int(w / resize_factor))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# blurr the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Identify circles
all_circles = cv2.HoughCircles(
    blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=0.1,
    minDist=int(100 / resize_factor),
    param1=int(20 / resize_factor),
    param2=int(10 / resize_factor),
    minRadius=int(115 / resize_factor),
    maxRadius=int(120 / resize_factor),
)
circles = np.uint16(np.around(all_circles))
print("It found " + str(circles.shape[1]) + " circles on the pi&d")

print(circles)

count = 0
img_circles = resized.copy()
for circle in circles[0]:
    # Annotate circle and centroid
    cv2.circle(img_circles, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    cv2.circle(img_circles, (circle[0], circle[1]), 2, (255, 0, 0), -2)

    # Annotate text
    offset_txt = int(circle[2] * 1.2)
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

# Read information in every circle
cropped_imgs = []
cropped_imgs_txt = []
img_circle_txt = img.copy()

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\diego.giraldo\AppData\Local\Tesseract-OCR\Tesseract.exe"
)

circles_int = circles[0] * resize_factor

for circle in circles_int:
    x_offset_right = np.uint16(circle[2] * 0.75)
    x_offset_left = np.uint16(circle[2] * 0.75)
    y_offset_low = np.uint16(circle[2] * 0.75)
    y_offset_up = np.uint16(circle[2] * 0.75)
    cropped_img_lower = img_circle_txt[
        circle[1] : circle[1] + y_offset_low,
        circle[0] - x_offset_left : circle[0] + x_offset_right,
    ]
    cropped_img_upper = img_circle_txt[
        circle[1] - y_offset_up : circle[1],
        circle[0] - x_offset_left : circle[0] + x_offset_right,
    ]
    #     cropped_img = cv2.threshold(cropped_img, 100, 255, cv2.THRESH_BINARY)
    cropped_imgs.append(np.append(cropped_img_upper, cropped_img_lower, axis=0))

    upper = pytesseract.image_to_string(
        cropped_img_upper,
        lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    lower = pytesseract.image_to_string(
        cropped_img_lower,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    r_upper = re.compile(r"[A-Z]+")
    r_lower = re.compile(r"[A-Z0-9]+")

    if r_upper.match(str(upper)) is not None:
        up = r_upper.match(str(upper)).group(0)

    if r_lower.match(str(lower)) is not None:
        low = r_lower.match(str(lower)).group(0)

    cropped_imgs_txt.append(up + "-" + low)


plt.imsave(fname="img_output.jpg", arr=img_circles)

for idx, c_img in enumerate(cropped_imgs):
    cv2.putText(
        c_img,
        cropped_imgs_txt[idx],
        (50, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        1,
    )
    cv2.imshow("Image", c_img)
    cv2.waitKey(0)
