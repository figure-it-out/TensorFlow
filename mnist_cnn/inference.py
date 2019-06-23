import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

def conv_layer(layer_name, input_tensor, conv_size, input_depth, output_depth):
    
    with tf.variable_scope(layer_name):
        weights = tf.get_variable("weight", [conv_size, conv_size, input_depth, output_depth], initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [output_depth], initializer = tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1,1,1,1], padding='SAME')
        result = tf.nn.relu(tf.nn.bias_add(conv, biases))
        
    return result

def normal_layer(layer_name, input_tensor, input_size, output_size, act=tf.nn.relu):
    
    with tf.variable_scope(layer_name):
        weights = tf.get_variable("weights", [input_size, output_size], initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [output_size], initializer = tf.constant_initializer(0.1))
        result = act(tf.matmul(input_tensor, weights) + biases)
        
    return result

def fun_inference(input_tensor):
    
    layer1_result = conv_layer('layer1-conv1', input_tensor, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP)
    
    with tf.name_scope('layer2-pool1'):
        layer2_result = tf.nn.max_pool(layer1_result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
    layer3_result = conv_layer('layer3-conv2', layer2_result, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP)
    
    with tf.name_scope('layer4-pool2'):
        layer4_result = tf.nn.max_pool(layer3_result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
    result_shape = layer4_result.get_shape().as_list()
    num_nodes = result_shape[1] * result_shape[2] * result_shape[3]
    
    reshaped = tf.reshape(layer4_result, [result_shape[0], num_nodes])
    
    layer5_result = normal_layer('layer5', reshaped, num_nodes, FC_SIZE)
    
    
    layer6_result = normal_layer('layer6', layer5_result, FC_SIZE, NUM_LABELS)
    
    return layer6_result
    
    
    
    
    
        
        
        
    
    
    
        
    
        
        