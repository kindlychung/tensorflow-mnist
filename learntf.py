import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.Graph().as_default():
    with tf.Session() as sess:
        my_var = tf.Variable(tf.truncated_normal([3, 4]), name="myvar", dtype=tf.float32)
        my_var_plus = my_var.assign_add(np.ones([3, 4], dtype=np.float32) * 10.0)
        writer = tf.summary.FileWriter("./tf_summary", sess.graph)
        init = tf.variables_initializer([my_var])
        sess.run(init)
        for _ in range(10):
            res = sess.run(my_var_plus)
        print(res)
        ##################################
        # Assign new value
        my_var_reassign = my_var.assign(tf.ones([3, 4]))
        res = sess.run(my_var_reassign)
        print(res)
        writer.close()

