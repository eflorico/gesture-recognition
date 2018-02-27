import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model_input import input_pipeline
from model import CNNModel, RNNModel, MaxPoolingModel
import sys

# Note that the evaluation code should have the same configuration.
config = {}
# Get from dataset.
config['num_test_samples'] = 2174
config['num_validation_samples'] = 1765
config['num_training_samples'] = 5722

config['batch_size'] = 16
config['learning_rate'] = 1e-3
# Learning rate is annealed exponentally in 'exponential' case. Don't forget to change annealing configuration in the code.
config['learning_rate_type'] = 'fixed' #'exponential'

config['num_steps_per_epoch'] = int(config['num_training_samples']/config['batch_size'])

config['num_epochs'] = 1000
config['evaluate_every_step'] = config['num_steps_per_epoch']
config['checkpoint_every_step'] = config['num_steps_per_epoch']
config['num_validation_steps'] = int(config['num_validation_samples']/config['batch_size'])
config['print_every_step'] = config['num_steps_per_epoch']
config['log_dir'] = './runs/'

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
# Maxpooling model parameters
config['maxpooling'] = {}
config['maxpooling']['num_hidden_units'] = [512]
config['maxpooling']['num_layers'] = 1
config['maxpooling']['num_class_labels'] = 20
config['maxpooling']['initializer'] = tf.contrib.layers.xavier_initializer()
config['maxpooling']['batch_size'] = config['batch_size']
config['maxpooling']['dropout_rate'] = 0.5

config['model'] = MaxPoolingModel
config['model_config'] = 'maxpooling'

config['ip_queue_capacity'] = config['batch_size']*50
config['ip_num_read_threads'] = 6

config['train_data_dir'] = "./train/"
config['train_file_format'] = "dataTrain_%d.tfrecords"
config['train_file_ids'] = list(range(1,41))
config['valid_data_dir'] = "./validation/"
config['valid_file_format'] = "dataValidation_%d.tfrecords"
config['valid_file_ids'] = list(range(1,16))

# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], timestamp))
print("Writing to {}\n".format(config['model_dir']))

# Create a list of tfRecord input files.
train_filenames = [os.path.join(config['train_data_dir'], config['train_file_format'] % i) for i in config['train_file_ids']]
# Create data loading operators. This will be represented as a node in the computational graph.
train_batch_samples_op, train_batch_labels_op, train_batch_seq_len_op = input_pipeline(train_filenames, config, name='training_input_pipeline', mode="training")

# Create a list of tfRecord input files.
valid_filenames = [os.path.join(config['valid_data_dir'], config['valid_file_format'] % i) for i in config['valid_file_ids']]
# Create data loading operators. This will be represented as a node in the computational graph.
valid_batch_samples_op, valid_batch_labels_op, valid_batch_seq_len_op = input_pipeline(valid_filenames, config, name='validation_input_pipeline', shuffle=False, mode="training")

# Create placeholders for training and monitoring variables.
loss_avg_op = tf.placeholder(tf.float32, name="loss_avg")
accuracy_avg_op = tf.placeholder(tf.float32, name="accuracy_avg")

# Generate a variable to contain a counter for the global training step.
# Note that it is useful if you save/restore your network.
global_step = tf.Variable(1, name='global_step', trainable=False)

# Create seperate graphs for training and validation.
# Training graph
# Note that our model is optimized by using the training graph.
with tf.name_scope("Training"):
    # Create model
    cnnModel = CNNModel(config=config['cnn'],
                        input_op=train_batch_samples_op, 
                        mode='training')
    cnn_representations = cnnModel.build_graph()
    
    trainModel = config['model'](config=config[config['model_config']], 
                            input_op=cnn_representations, 
                            target_op=train_batch_labels_op, 
                            seq_len_op=train_batch_seq_len_op,
                            mode="training")
    trainModel.build_graph()
    print("\n# of parameters: %s" % trainModel.num_parameters)
    
    # Optimization routine.
    # Learning rate is decayed in time. This enables our model using higher learning rates in the beginning.
    # In time the learning rate is decayed so that gradients don't explode and training staurates.
    # If you observe slow training, feel free to modify decay_steps and decay_rate arguments.
    if config['learning_rate_type'] == 'exponential':
        learning_rate = tf.train.exponential_decay(config['learning_rate'], 
                                                   global_step=global_step,
                                                   decay_steps=1000, 
                                                   decay_rate=0.97,
                                                   staircase=False)
    elif config['learning_rate_type'] == 'fixed':
        learning_rate = config['learning_rate']
    else:
        print("Invalid learning rate type")
        raise
        
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(trainModel.loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# Validation graph.
with tf.name_scope("Evaluation"):
    # Create model
    validCnnModel = CNNModel(config=config['cnn'],
                                input_op=valid_batch_samples_op, 
                                mode='validation')
    valid_cnn_representations = validCnnModel.build_graph()
    
    validModel = config['model'](config=config[config['model_config']],
                            input_op=valid_cnn_representations, 
                            target_op=valid_batch_labels_op, 
                            seq_len_op=valid_batch_seq_len_op,
                            mode="validation")
    validModel.build_graph()
    
# Create summary ops for monitoring the training.
# Each summary op annotates a node in the computational graph and collects
# data data from it.
summary_train_loss = tf.summary.scalar('loss', trainModel.loss)
summary_train_acc = tf.summary.scalar('accuracy_training', trainModel.batch_accuracy)
summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg_op)
summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg_op)
summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)

# Group summaries.
# summaries_training is used during training and reported after every step.
summaries_training = tf.summary.merge([summary_train_loss, summary_train_acc, summary_learning_rate])
# summaries_evaluation is used by both trainig and validation in order to report the performance on the dataset.
summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])
    
#Create session object
sess = tf.Session()
# Add the ops to initialize variables.
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# Actually intialize the variables
sess.run(init_op)

# Register summary ops.
train_summary_dir = os.path.join(config['model_dir'], "summary", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
valid_summary_dir = os.path.join(config['model_dir'], "summary", "validation")
valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)

# Define counters in order to accumulate measurements.
counter_correct_predictions_training = 0.0
counter_loss_training = 0.0
counter_correct_predictions_validation = 0.0
counter_loss_validation = 0.0

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Training Loop
try:
    while not coord.should_stop():
        step = tf.train.global_step(sess, global_step)
            
        if (step%config['checkpoint_every_step']) == 0:
            ckpt_save_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
            print("Model saved in file: %s" % ckpt_save_path)
            sys.stdout.flush()
            
        # Run the optimizer to update weights.
        # Note that "train_op" is responsible from updating network weights.
        # Only the operations that are fed are evaluated.
        # Run the optimizer to update weights.
        train_summary, num_correct_predictions, loss, _ = sess.run([summaries_training, 
                                                                      trainModel.num_correct_predictions, 
                                                                      trainModel.loss, 
                                                                      train_op], 
                                                                      feed_dict={})
        # Update counters.
        counter_correct_predictions_training += num_correct_predictions
        counter_loss_training += loss
        # Write summary data.
        train_summary_writer.add_summary(train_summary, step)
        
        # Report training performance
        if (step%config['print_every_step']) == 0 or step == 1:
            accuracy_avg = counter_correct_predictions_training / (config['batch_size']*config['print_every_step'])
            loss_avg = counter_loss_training / (config['print_every_step'])
            summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg_op:accuracy_avg, loss_avg_op:loss_avg})
            train_summary_writer.add_summary(summary_report, step)
            print("[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f" % (step/config['num_steps_per_epoch'], 
                                                                     step, 
                                                                     accuracy_avg, 
                                                                     loss_avg))
            sys.stdout.flush()
            counter_correct_predictions_training = 0.0
            counter_loss_training= 0.0
        
        if (step%config['evaluate_every_step']) == 0:
            # It is possible to create only one input pipelene queue. Hence, we create a validation queue 
            # in the begining for multiple epochs and control it via a foor loop.
            # Note that we only approximate 1 validation epoch (validation doesn't have to be accurate.)
            # In other words, number of unique validation samples may differ everytime.
            for eval_step in range(config['num_validation_steps']):
                # Calculate average validation accuracy.
                num_correct_predictions, loss = sess.run([validModel.num_correct_predictions, 
                                                          validModel.loss],
                                                         feed_dict={})
                # Update counters.
                counter_correct_predictions_validation += num_correct_predictions
                counter_loss_validation += loss
            
            # Report validation performance
            accuracy_avg = counter_correct_predictions_validation / (config['batch_size']*config['num_validation_steps'])
            loss_avg = counter_loss_validation / (config['num_validation_steps'])
            summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg_op:accuracy_avg, loss_avg_op:loss_avg})
            valid_summary_writer.add_summary(summary_report, step)
            print("[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f" % (step/config['num_steps_per_epoch'], 
                                                                       step, 
                                                                       accuracy_avg, 
                                                                       loss_avg))
            sys.stdout.flush()
            counter_correct_predictions_validation = 0.0
            counter_loss_validation= 0.0
        
except tf.errors.OutOfRangeError:
    print('Model is trained for %d epochs, %d steps.' % (config['num_epochs'], step))
    print('Done.')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)

ckpt_save_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
print("Model saved in file: %s" % ckpt_save_path)
sess.close()
