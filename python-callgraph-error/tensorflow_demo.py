import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def train_test():
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]
    y_train = to_categorical(y_train, num_classes=10)  # Convert labels to one-hot encoding
    y_test = to_categorical(y_test, num_classes=10)

    # Create a sequential model
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images to a 1D vector
        Dense(128, activation='relu'),   # Fully connected layer with 128 units and ReLU activation
        Dense(10, activation='softmax')  # Fully connected layer with 10 units for 10 classes and softmax activation
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_accuracy)

def build_graph(test=False):
	with tf.name_scope('imgholder'): # The placeholder is just a holder and doesn't contains the actual data.
		imgholder = tf.placeholder(tf.float32,[None,256,256,3]) # The 3 is color channels
	with tf.name_scope('bias_holder'):
		bias_holder = tf.placeholder(tf.float32,[None,16,16,4]) # The bias (x,y,w,h) for 16*16 feature maps.
	with tf.name_scope('conf_holder'):
		conf_holder = tf.placeholder(tf.float32,[None,16,16,1]) # The confidence about 16*16 feature maps.
	with tf.name_scope('croppedholder'):
		croppedholder = tf.placeholder(tf.float32,[None,32,32,3]) # 256 is the number of feature maps
	with tf.name_scope('veri_conf_holder2'):
		veri_conf_holder = tf.placeholder(tf.float32, [None,1])
#	with tf.name_scope('veri_bias_holder'):
#		veri_bias_holder = tf.placeholder(tf.float32, [None,4]) # The veri output numbers,x,y,w,h

	with tf.name_scope('mask'):
		maskholder = tf.placeholder(tf.float32,[None,16,16,1])

	conf, bias,feature_map = RPN(imgholder,test)
	veri_conf = verify_net(croppedholder,test)
	
	bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias*conf_holder - bias_holder),axis=0))
	conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conf,labels=conf_holder))

def ResourceScatterUpdate_demo():
    v = tf.Variable([b'vvv'])
    tf.raw_ops.ResourceScatterUpdate(
    resource=v.handle,
    indices=[0],
    updates=['1', '2', '3', '4', '5'])

def UncompressElementDemo():
    data = tf.data.Dataset.from_tensors([0.0])
    tf.raw_ops.UncompressElement(
    compressed=tf.data.experimental.to_variant(data),
    output_types=[tf.int64],
    output_shapes=[2])

def ReverseSequenceDemo():
    y = tf.raw_ops.ReverseSequence(
    input = ['aaa','bbb'],
    seq_lengths = [1,1,1],
    seq_dim = -10,
    batch_dim = -10 )
    return y

def EditDistanceDemo():
    hypothesis_indices = tf.constant(-1250999896764, shape=[3, 3], dtype=tf.int64) 
    hypothesis_values = tf.constant(0, shape=[3], dtype=tf.int64)
    hypothesis_shape = tf.constant(0, shape=[3], dtype=tf.int64)
    truth_indices = tf.constant(-1250999896764, shape=[3, 3], dtype=tf.int64)
    truth_values = tf.constant(2, shape=[3], dtype=tf.int64)
    truth_shape = tf.constant(2, shape=[3], dtype=tf.int64) 
   
    tf.raw_ops.EditDistance(
    hypothesis_indices=hypothesis_indices,
    hypothesis_values=hypothesis_values,
    hypothesis_shape=hypothesis_shape,
    truth_indices=truth_indices,
    truth_values=truth_values,
    truth_shape=truth_shape
    )
    

if __name__ == '__main__':
    #build_graph()
    train_test()
    ResourceScatterUpdate_demo()
    UncompressElementDemo()
    ReverseSequenceDemo()
    EditDistanceDemo()