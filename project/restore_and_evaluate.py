import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model_input import  input_pipeline
from model import CNNModel, RNNModel

config = {}
# Get from dataset.
config['num_test_samples'] = 2174
config['batch_size'] = 16

config['num_epochs'] = 1
config['model_dir'] = './runs/1497551635/'
config['checkpoint_id'] = None # If None, the last checkpoint will be used.

config['img_height'] = 80
config['img_width'] = 80
config['img_num_channels'] = 3
config['skeleton_size'] = 180

# CNN model parameters
config['cnn'] = {}
config['cnn']['cnn_filters'] = [16,32,64,128] # Number of filters for every convolutional layer.
config['cnn']['num_hidden_units'] = 512 # Number of output units, i.e. representation size.
config['cnn']['dropout_rate'] = 0.5
config['cnn']['initializer'] = tf.contrib.layers.xavier_initializer()
# RNN model parameters
config['rnn'] = {}
config['rnn']['num_hidden_units'] = 128 # Number of units in an LSTM cell.
config['rnn']['num_layers'] = 1 # Number of LSTM stack.
config['rnn']['num_class_labels'] = 20
config['rnn']['initializer'] = tf.contrib.layers.xavier_initializer()
config['rnn']['batch_size'] = config['batch_size']
config['rnn']['loss_type'] = 'average' # or 'last_step' # In the case of 'average', average of all time-steps is used instead of the last time-step.

config['ip_queue_capacity'] = config['batch_size']*50
config['ip_num_read_threads'] = 1

config['test_data_dir'] = "../test/"
config['test_file_format'] = "dataTest_%d.tfrecords"
config['test_file_ids'] = list(range(1,16))

# Create a list of tfRecord input files.
test_filenames = [os.path.join(config['test_data_dir'], config['test_file_format'] % i) for i in config['test_file_ids']]
# Create data loading operators. This will be represented as a node in the computational graph.
test_batch_samples_op, test_batch_ids_op, test_batch_seq_len_op = input_pipeline(test_filenames, config, name='test_input_pipeline', shuffle=False, mode="inference")

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Test graph.
with tf.name_scope("Inference"):
    # Create model
    inferCnnModel = CNNModel(config=config['cnn'],
                                input_op=test_batch_samples_op, 
                                mode='inference')
    infer_cnn_representations = inferCnnModel.build_graph()
    
    inferModel = RNNModel(config=config['rnn'], 
                            input_op=infer_cnn_representations, 
                            target_op=None, 
                            seq_len_op=test_batch_seq_len_op,
                            mode="inference")
    inferModel.build_graph()
    
# Restore computation graph.
saver = tf.train.Saver()
# Restore variables.
checkpoint_path = config['checkpoint_id']
if checkpoint_path is None:
    checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
print("Evaluating " + checkpoint_path)
saver.restore(sess, checkpoint_path)


# Evaluation loop
test_predictions = []
test_sample_ids = []
try:
    while not coord.should_stop():
        # Get predicted labels and sample ids for submission csv.
        [predictions, sample_ids] = sess.run([inferModel.predictions, test_batch_ids_op], feed_dict={})
        test_predictions.extend(predictions)
        test_sample_ids.extend(sample_ids)

except tf.errors.OutOfRangeError:
    print('Done.')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()   

# Wait for threads to finish.
coord.join(threads)

# Now you have your predictions. Do whatever you want:
f = open('submission.csv','w')
print('Id,Prediction', file=f)
for id, pred in zip(test_predictions, test_sample_ids):
    print('%d,%d' % (id, pred+1), file=f)
f.close()