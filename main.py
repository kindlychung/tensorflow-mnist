import numpy as np
from PIL import Image
import os
import requests
import numpy as np
from io import BytesIO
from flask import Flask, jsonify, render_template, request
import tensorflow as tf

from mnist.util.preprocess import preprocess

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

regression_checkpoint = os.path.join(os.path.dirname(__file__), 'mnist', 'data', 'regression.ckpt')
# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, regression_checkpoint)


convnet_checkpoint = os.path.join(os.path.dirname(__file__), 'mnist', 'data', 'convolutional1.ckpt')
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, convnet_checkpoint)


def regression(input):
    print(input.shape)
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    print(input.shape)
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = np.array(request.json, dtype=np.uint8).reshape((28, 28))
    probs1 = regression(preprocess(input))
    _, probs2 = predict(input)
    return jsonify(results=[probs1, probs2])


def predict(input):
    input = preprocess(input)
    probs = convolutional(input)
    pred = np.argmax(probs)
    return (pred, probs)

@app.route('/api/convnet/imgraw', methods=['POST'])
def convnet():
    input = np.array(request.json, dtype=np.double).reshape(28, 28)
    pred, probs = predict(input)
    return jsonify(pred=str(pred), probs=probs)

@app.route('/api/convnet/imgurl', methods=['POST'])
def convnet_from_url():
    # url = "http://i.imgur.com/76oiYoJ.png"
    url = request.json[0]
    response = requests.get(url)
    input = Image.open(BytesIO(response.content)).convert("L")
    pred, probs = predict(input)
    return jsonify(pred=str(pred), probs=probs)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()


