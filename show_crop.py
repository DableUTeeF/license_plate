import cv2
from PIL import Image
image = cv2.imread('/home/palmy/PycharmProjects/DATA/Tanisorn/imgCarResize/CAR0112.jpg')
# ymin, ymax, xmin, xmax
image = cv2.rectangle(image, (67, 311), (145, 375), (0, 255, 0), 3)
# image = cv2.rgbd
im = Image.fromarray(image)
im.show()
