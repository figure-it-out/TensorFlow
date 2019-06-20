import tensorflow as tf

INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10



def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):

    with tf.variable_scope(layer_name):
        weights = tf.get_variable('weights', [input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.summary.histogram(layer_name + '/weights', weights)
        
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(layer_name + 'biases', biases)
        
        output = act(tf.matmul(input_tensor, weights) + biases)

    return output

def fun_inference(input_tensor):
    hidden1 = nn_layer(input_tensor, INPUT_NODE, LAYER1_NODE, 'layer1')
    y = nn_layer(hidden1, LAYER1_NODE, OUTPUT_NODE, 'layer2', act=tf.identity)
    return y    
        
def writer():        
    writer = tf.summary.FileWriter("log",tf.get_default_graph())
    writer.close()