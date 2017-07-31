import cv2
import os
import uuid
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave


def rand(limit):
    return np.random.uniform(.01, limit, size=1)[0]


def rand_perspective(four_points, max_w, max_h, limit):
    assert len(four_points) == 4, "You should give 4 xy coordinates"
    assert 0 < limit < .5, "Limit should be 0-0.5"
    p1, p2, p3, p4 = copy.deepcopy(four_points)
    p1x = p1[0] + max_w * rand(limit)
    p1y = p1[1] + max_h * rand(limit)
    p2x = p2[0] - max_w * rand(limit)
    p2y = p2[1] + max_h * rand(limit)
    p3x = p3[0] - max_w * rand(limit)
    p3y = p3[1] - max_h * rand(limit)
    p4x = p4[0] + max_w * rand(limit)
    p4y = p4[1] - max_h * rand(limit)
    return np.array([
        [p1x, p1y],
        [p2x, p2y],
        [p3x, p3y],
        [p4x, p4y]
    ], dtype=np.float32)


def warpImage(img):
    h, w = img.shape[:2]
    rect = np.array([
        [0, 0], [w, 0],
        [w, h], [0, h]
    ], dtype=np.float32)
    dst = rand_perspective(rect, w, h, .2)
    M = cv2.getPerspectiveTransform(rect, dst)
    scale = np.random.uniform(0.7, 1.5)
    w1 = int(w * scale)
    h1 = int(h * scale)
    warp = cv2.warpPerspective(img, M, (w, h))
    warp = cv2.resize(warp, (w1, h1))
    return warp


def resize_background(img, side_len):
    h, w = img.shape[:2]
    if w < h:
        w1 = side_len
        h1 = int(h / w * w1)
    else:
        h1 = side_len
        w1 = int(w / h * h1)
    img = cv2.resize(img, (w1, h1))
    return img[:side_len, :side_len, :]


def makeYellow(transparency=None):
    r = np.random.choice(np.arange(188, 255))
    g = 111
    b = 54
    if transparency is None:
        return (r, g, b)
    else:
        return (r, g, b, transparency)


def makeSize(string, fontSize):
    h = int(fontSize * 1.1)
    w = int(h / 1.65) * len(string)
    return [w, h]


def warpText(label, fontSize):
    fonts = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    ]
    font = np.random.choice(fonts, 1)[0]
    fnt = ImageFont.truetype(font, fontSize)
    txt = Image.new('RGBA', makeSize(label, fontSize), makeYellow(255))
    # get a drawing context
    d = ImageDraw.Draw(txt)
    # draw text, half opacity
    d.text((0, 0), label, font=fnt, fill=(0, 0, 0, 255))
    txt = warpImage(np.array(txt))
    return txt


def translateIndex(idx, dx, dy):
    return (idx[0] + dy, idx[1] + dx)


def compose(background, txt, label):
    h1, w1 = txt.shape[:2]
    h2, w2 = background.shape[:2]
    scopeX = w2 - w1
    scopeY = h2 - h1
    assert scopeX > 0 and scopeY > 0, "Background image should be bigger than insert!"
    dx = int(np.random.uniform(0, scopeX))
    dy = int(np.random.uniform(0, scopeY))
    idx = np.nonzero(txt[:, :, 3])
    background[translateIndex(idx, dx, dy)] = txt[idx]
    # centerX = int(dx + (w1 / 2))
    # centerY = int(dy + (h1 / 2))
    return {
        "image": background,
        "bounds": [
            [dx, dy, dx + w1, dy + h1]
        ],
        "labels": [label]
    }


def compose_multple(background, txts, labels):
    assert len(txts) > 0, "Inserts cannot be empty!"
    res = compose(background, txts[0], labels[0])
    for txt, label in zip(txts[1:], labels[1:]):
        tmp = compose(background, txt, label)
        res["image"] = tmp["image"]
        res["bounds"].append(tmp["bounds"][0])
        res["labels"].append(tmp["labels"][0])
    return res


def xmlSize(img):
    h, w = img.shape[:2]
    size = etree.Element("size")
    width = tag_with_text("width", str(w))
    height = tag_with_text("height", str(h))
    depth = tag_with_text("depth", "3")
    size.append(width)
    size.append(height)
    size.append(depth)
    return size


def tag_with_text(tag_name, text):
    tag = etree.Element(tag_name)
    tag.text = text
    return tag


def createObj(label, bounds, pose_type="Unspecified", truncated_type="0", difficult_type="0"):
    obj = etree.Element("object")
    name = tag_with_text("name", label)
    pose = tag_with_text("pose", pose_type)
    truncated = tag_with_text("truncated", truncated_type)
    difficult = tag_with_text("difficult", difficult_type)
    ###########
    # bounding-box
    x1, y1, x2, y2 = bounds
    xmin = tag_with_text("xmin", str(x1))
    ymin = tag_with_text("ymin", str(y1))
    xmax = tag_with_text("xmax", str(x2))
    ymax = tag_with_text("ymax", str(y2))
    bndbox = etree.Element("bndbox")
    for e in [xmin, ymin, xmax, ymax]:
        bndbox.append(e)
    ########### end bounding-box
    for e in [name, pose, truncated, difficult, bndbox]:
        obj.append(e)
    return obj


def create_annotation(sample, file_name, output):
    img = sample["image"]
    bounds = sample["bounds"]
    labels = sample["labels"]
    folder_name = "Images"
    root = etree.Element("annotation")
    folder = tag_with_text("folder", folder_name)
    filename = tag_with_text("filename", file_name)
    size = xmlSize(img)
    segmented = tag_with_text("segmented", "0")
    objs = list(map(lambda x: createObj(*x), zip(labels, bounds)))
    for e in [folder, filename, size, segmented] + objs:
        root.append(e)
    with open(output, "bw") as fh:
        fh.write(etree.tostring(root, pretty_print=True))


class YoloTrainConfig:
    def __init__(self, label_pool, bg_folder, annotation_folder, images_folder, bg_size):
        self.label_pool = label_pool
        self.bg_folder = bg_folder
        self.annotation_folder = annotation_folder
        self.images_folder = images_folder
        self.bg_size = bg_size
        os.makedirs(self.annotation_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        self.bg_pool = os.listdir(self.bg_folder)

    def rand_bg(self):
        bg_file = os.path.join(
            self.bg_folder,
            np.random.choice(self.bg_pool, 1)[0]
        )
        return bg_file

    def rand_labels(self, n=3):
        return np.random.choice(self.label_pool, n, replace=True)

    def rand_filepaths(self):
        file_root = str(uuid.uuid4()).replace("-", "")
        png_file = file_root + ".png"
        png_path = os.path.join(self.images_folder, png_file)
        xml_file = file_root + ".xml"
        xml_path = os.path.join(self.annotation_folder, xml_file)
        return png_path, xml_path



config = YoloTrainConfig(
    label_pool=["5004", "5003", "23", "24"],
    bg_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/bg_images",
    annotation_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/yolo_train_data/annotations",
    images_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/yolo_train_data/images",
    bg_size=416
)


def make_annotated_img(config: YoloTrainConfig):
    labels = config.rand_labels()
    png_path, xml_path = config.rand_filepaths()
    bg_file = config.rand_bg()
    background = resize_background(
        np.array(
            Image.open(bg_file).convert("RGBA")
        ),
        config.bg_size
    )
    txts = list(map(lambda x: warpText(x, 40), labels))
    out = compose_multple(background, txts, labels)
    # cv2.imwrite(png_path, out["image"])
    imsave(png_path, out["image"])
    create_annotation(out, os.path.basename(png_path), xml_path)


for i in range(1000):
    make_annotated_img(config)



config_test = YoloTrainConfig(
    label_pool=["5004", "5003", "23", "24", "500"],
    bg_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/bg_images",
    annotation_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/yolo_test_data/annotations",
    images_folder="/home/kaiyin/PycharmProjects/tensorflow-mnist/yolo_test_data/images",
    bg_size=416
)

for i in range(300):
    make_annotated_img(config_test)

