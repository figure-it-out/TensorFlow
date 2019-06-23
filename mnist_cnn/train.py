from termcolor import colored
from datetime import datetime

import tensorflow as tf
import numpy as np

import inference
from tensorflow.examples.tutorials.mnist import input_data

REGULARIZATION = False
LEARNING_RATE_BASE = 0.6
LEARNING_DECAY = 0.99
BATCH_SIZE = 100
TRAINING_STEPS = 100000
LOG_DIR = "log/"

def train(mnist):
    
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS],
        name='x-input')
    
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
    
    y = inference.fun_inference(x)
    
    global_step = tf.Variable(0, trainable=False, name = 'global_step')
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean  = tf.reduce_mean(cross_entropy)
    
    loss = cross_entropy_mean
 
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_DECAY)
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    merged = tf.summary.merge_all()
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    print(colored("training begins", 'red'))
    with tf.Session() as sess:
        log_dir = LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        for i in range(TRAINING_STEPS):
            x_m,y_m = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xm = np.reshape(x_m, (
                BATCH_SIZE,
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS))
            summary, no_use, loss_value= sess.run([merged, train_step, loss], feed_dict={x:reshaped_xm, y_:y_m})
            
            writer.add_summary(summary, i)
            
            if i % 1000 == 0:
                print ("step: %d, loss_value:%g" % (i, loss_value))
                
    writer.close()
    
def main(argv=None):
    print(colored("data reading begins", 'red'))
    mnist = input_data.read_data_sets("data/", one_hot = True)
    print(colored("data reading ends", 'red'))
    train(mnist)
if __name__ == '__main__':
    tf.app.run()