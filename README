# User Interface Engineering

*Project work for the course User Interface Engineering by Prof. Ottmar Hilliges.*

We were provided with a modified version of the [ChaLearn][1] dataset for gesture recognition. In assignment 2, we trained a convolutional neural network to recognize gestures from individual frames. In the final project, we trained a neural network on the full movie clips. We evaluated multiple architectures to incorporate the temporal information: either combining features learned by a CNN with an RNN or maxpooling the CNN features. 

In both cases, it proved crucial to augment the training dataset as much as possible. To that end, we made sure our training pipeline included random rotations, crops, channel shifts, changes in brightness and contrast, and flipped images horizontally. 

Assignment 2 was completed using Keras, while the final project was written in native TensorFlow, as the large amount of data made it necessary to fully control the input pipeline and its memory consumption.

## Results
  - Assignment 2: 1st out of 33 students; Accuracy: 0.878
  - Final project: 3rd out of 15 teams; Accuracy: 0.775

[1]: http://chalearnlap.cvc.uab.es