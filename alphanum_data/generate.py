import numpy as np
import string
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
from matplotlib.image import imsave

template = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" width="13mm" height="13mm" viewBox="0 0 46.062992 46.06299" id="svg2" version="1.1">
    <defs id="defs4">
        <filter style="color-interpolation-filters:sRGB" id="filter1025" x="-0.21332018" width="1.4266404" y="-0.19546016" height="1.3909203">
            <feGaussianBlur stdDeviation="{sd}" id="feGaussianBlur1027" />
        </filter>
    </defs>
    <metadata id="metadata7">
        <rdf:RDF>
            <cc:Work rdf:about="">
                <dc:format>image/svg+xml</dc:format>
                <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
                <dc:title></dc:title>
            </cc:Work>
        </rdf:RDF>
    </metadata>
    <g id="layer1" transform="translate(-62.285296,-37.714502)">
        <text xml:space="preserve" style="font-style:{style};font-weight:normal;line-height:0%;font-family:{font};letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;filter:url(#filter1025)" x="71.215752" y="74.534538" id="text10">
            <tspan id="tspan12" x="71.215752" y="74.534538" style="font-size:40px;line-height:1.25">{text}</tspan>
        </text>
    </g>
</svg>
"""


def add_noise(input, output_basename):
    for noise_mode in noise_mode_list:
        img = np.array(Image.open(input))
        img = random_noise(img, mode=noise_mode) * 255 + img
        img = random_noise(img, mode=noise_mode) * (-255) + img
        img = np.absolute(img)
        img = (img / img.max() * 255).astype("uint8")
        img = Image.fromarray(img)
        img.save("{}_{}.png".format(output_basename, noise_mode))

sd_list = np.linspace(0, 2, 10)
style_list = ["normal", "italic"]
font_list = ["Monospace", "Liberation Mono", "Loma", "Linux Libertine O", "Liberation Serif", "Linux Biolinum O", "Kinnari", "DejaVu Serif", "Noto Sans", "Noto Sans Mono"]
text_list = string.ascii_letters + "0123456789"
noise_mode_list = ['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
# data_dir = "/home/kaiyin/PycharmProjects/tensorflow-mnist/alphanum_data"
data_dir = "/tmp"

for text in text_list[:1]:
    for sd in sd_list:
        for style in style_list[:1]:
            for font in font_list[:2]:
                params = {"sd": sd, "style": style, "font": font, "text": text}
                wd = os.path.join(data_dir, text)
                if not os.path.exists(wd):
                    os.mkdir(wd)
                output_basename = "{text}_{font}_{style}_{sd}".format(**params)
                svg_filename = os.path.join(wd, "{}.svg".format(output_basename))
                png_filename = os.path.join(wd, "{}.png".format(output_basename))
                svg = template.format(**params)
                with open(svg_filename, "w") as fh:
                    fh.write(svg)
                os.system("convert '{}' -set colorspace Gray -resize 48x48^ '{}'".format(svg_filename, png_filename))
                os.remove(svg_filename)
                noised_basename = os.path.join(wd, output_basename)
                add_noise(png_filename, noised_basename)





add_noise("/media/kaiyin/wdDataTransfer/alphanum_data/a/a_Liberation Mono_italic_0.0.png", "/tmp/x")
tmp_files = ["/tmp/x_{}.png".format(m) for m in noise_mode_list]
imgs = [np.array(Image.open(f)) for f in tmp_files]
plt.imshow(imgs[0])
plt.imshow(imgs[1])
plt.imshow(imgs[2])
plt.imshow(imgs[3])
plt.imshow(imgs[4])
plt.imshow(imgs[5])
plt.hist(imgs[0])
