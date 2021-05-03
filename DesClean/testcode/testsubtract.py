import tensorflow as tf
import matplotlib.pyplot as plt

#F = A + (1-A) * (1-B)**5
matrixA = tf.ones([256,256,3])*0.8
matrixB = tf.zeros([256,256,1])

A = 1 - matrixA
B = (1 - matrixB)**5


out = tf.math.multiply(A,B)

print(matrixA.shape,matrixB.shape)
print(out)


