import cv2

def read_img(filepath):
    return cv2.imread(filepath, cv2.CV_8UC1)

def blur_img(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def resize_img(img):
    w = 28
    h = 28
    b = blur_img(img)
    return cv2.resize(b, (w, h))