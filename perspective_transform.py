import cv2
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
import copy
import matplotlib.pyplot as plt
import numpy as np


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


background = np.array(Image.open("/home/kaiyin/PycharmProjects/tensorflow-mnist/moonwalk-mj.jpg").convert("RGBA"))
labels = ["5004", "5007", "24", "21"]
txts = list(map(lambda x: warpText(x, 40), labels))
plt.cla(); plt.imshow(txts[0])
plt.cla(); plt.imshow(txts[1])
plt.cla(); plt.imshow(txts[2])
out = compose_multple(background, txts, labels)
plt.imshow(out["image"])
print(out["bounds"])
print(out["labels"])




def xmlSize(img):
    h, w = img.shape[:2]
    size = etree.Element("size")
    width = etree.Element("width")
    width.text = str(w)
    height = etree.Element("height")
    height.text = str(h)
    depth = etree.Element("depth")
    depth.text = "3"
    size.append(width)
    size.append(height)
    size.append(depth)
    return size

img = background
folder_name = "Images"
file_idx = 1
file_root = "%05d" % file_idx
file_name = file_root + ".jpg"
root = etree.Element("annotation")
folder = etree.Element("folder")
folder.text = folder_name
filename = etree.Element("filename")
filename.text = file_name
root.append(folder)
root.append(filename)
size = xmlSize(img)
root.append(size)
segmented = etree.Element("segmented")
segmented.text = "0"
root.append(segmented)
print(etree.tostring(root, pretty_print=True).decode("utf8"))

xmlString = """<annotation>
  <object>
    <name>person</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>27</xmin>
      <ymin>42</ymin>
      <xmax>205</xmax>
      <ymax>375</ymax>
    </bndbox>
  </object>
  <object>
    <name>person</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>150</xmin>
      <ymin>31</ymin>
      <xmax>362</xmax>
      <ymax>373</ymax>
    </bndbox>
  </object>
  <object>
    <name>person</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>366</xmin>
      <ymin>57</ymin>
      <xmax>500</xmax>
      <ymax>311</ymax>
    </bndbox>
  </object>
  <object>
    <name>bottle</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>369</xmin>
      <ymin>163</ymin>
      <xmax>429</xmax>
      <ymax>333</ymax>
    </bndbox>
  </object>
  <object>
    <name>bottle</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>229</xmin>
      <ymin>178</ymin>
      <xmax>281</xmax>
      <ymax>286</ymax>
    </bndbox>
  </object>
</annotation>
"""
