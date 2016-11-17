import tensorflow as tf
import numpy as np
from tensorflow.python.training import optimizer

num_lex = 4
num_dc = 2
num_rst = 3
num_pdtb = 2
lex = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
sdc = lex[:, 0]
sdc.shape = (num_lex, 1)

sdc = sdc == np.arange(num_dc)

src = lex[:, 1]
src.shape = (num_lex, 1)
src = src == np.arange(num_rst)
sdc = sdc.astype(np.float32)
src = src.astype(np.float32)

e = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
sdc_e = np.dot(sdc, e)

m_logits = tf.Variable(np.random.rand(num_pdtb, num_rst).astype(np.float32))
m = tf.nn.softmax(m_logits)

test = tf.matmul(m, src.T)
score = tf.matmul(sdc_e, tf.matmul(m, src.T))
log_score = tf.log(score)
sum_log_score = tf.trace(log_score)
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(-sum_log_score)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(tf.neg(sum_log_score))


# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
print(sess.run(m))
print(sess.run(sum_log_score))

# Fit the line.
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
for step in range(100000):
    sess.run(optimizer)
    if step % 1000 == 0:
        print(step, sess.run(m), sess.run(sum_log_score))

