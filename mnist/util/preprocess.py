
from PIL import Image, ImageFilter
import math
import numpy as np

def resize(im):
    im = im.convert("L")
    width = im.size[0]
    height = im.size[1]
    new_img = Image.new("L", (28, 28), 255)
    if width > height:
        # h1/h = w1/w = 20/w
        new_height = math.ceil(20.0 * height / width)
        if (new_height == 0):
            new_height = 1
        img = im.resize((20, new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        top_margin = math.ceil((28 - new_height) / 2.0)
        new_img.paste(img, (4, top_margin))
    else :
        # h1/h = w1/w = 20/h
        new_width = math.ceil(20.0 * width / height)
        if (new_width == 0):
            new_width = 1
        img = im.resize((new_width, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        print("Size of new img: ", img.size)
        left_margin = math.ceil((28 - new_width) / 2.0)
        new_img.paste(img, (left_margin, 4))
    return new_img


def crop(img):
    image_data = np.asarray(img)
    threshold = 200
    non_empty_cols = np.where(image_data.min(axis=0) < threshold)[0]
    non_empty_rows = np.where(image_data.min(axis=1) < threshold)[0]
    row0 = min(non_empty_rows)
    row1 = max(non_empty_rows) + 1
    col0 = min(non_empty_cols)
    col1 = max(non_empty_cols) + 1
    image_data_new = image_data[row0:row1, col0:col1]
    return Image.fromarray(image_data_new)


def preprocess(img):
    img = resize(crop(img))
    img_data = np.asarray(img)
    return ((255.0 - img_data) / 255.0).reshape(1, 784)


