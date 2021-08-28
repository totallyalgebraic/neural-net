import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Removing warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


data_set = input_data.read_data_sets("data_set_data/", one_hot=True) 


# Important constants
num_in = 784  
num_layer1 = 512  
num_layer2 = 256  
num_layer3 = 128  
num_output = 10  


learning_rate = 1e-4
num_iterations = 1000
batch_size = 128
dropout = 0.5


X = tf.placeholder("float", [None, num_in])
Y = tf.placeholder("float", [None, num_output])
keep_prob = tf.placeholder(tf.float32)


weights = {
    'w1': tf.Variable(tf.truncated_normal([num_in, num_layer1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([num_layer1, num_layer2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([num_layer2, num_layer3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([num_layer3, num_output], stddev=0.1)),
}


biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[num_layer1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[num_layer2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[num_layer3])),
    'out': tf.Variable(tf.constant(0.1, shape=[num_output]))
}


layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(num_iterations):
    batch_x, batch_y = data_set.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
    })

    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), 
        "\t| Accuracy =", str(minibatch_accuracy))


test_accuracy = sess.run(accuracy, feed_dict={X: data_set.test.images, Y: data_set.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)