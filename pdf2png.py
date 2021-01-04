from pdf2image import convert_from_path
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None
output = "images_tests/"


def convert(file, output):
    if not os.path.exists(output):
        os.makedirs(output)

    pages = convert_from_path(file, 500)
    counter = 1

    for page in pages:
        my_file = output + "output" + str(counter) + ".jpg"
        counter += 1
        page.save(my_file, "JPEG")
        print(my_file)


file = (
    r"C:\Users\diego.giraldo\OneDrive - LOGICAMMS LTD\Downloads\05-P-PID-0253_0_1.pdf"
)

convert(file, output)