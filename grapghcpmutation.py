#computing gradient
import tensorflow as tf

x = tf.Variable(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx)
