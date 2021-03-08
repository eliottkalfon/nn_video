# Visualising a Neural Network's training process

This video shows the training of a neural network on a noisy and non-linearly separable dataset. The left chart shows the neurons and connections between them. Red dotted lines indicate large weight changes (learning). The right chart visually displays the evolution of the decision boundaries over the training epochs. Scroll down for a longer explanation and a description of the code implementation.

Click on the GIF to go to the mp4 video.

[![NN GIF](https://github.com/eliottkalfon/nn_video/blob/main/nn_gif.gif)](https://github.com/eliottkalfon/nn_video/blob/main/movie.mp4)

Notebook and code: https://github.com/eliottkalfon/nn_video/tree/main/code

Inspiration: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

## Description

A neural network is a set of connected neurons, usually arranged in layers. Each neuron is a single unit, passing a linear combination of its input signals through an activation function, and sending this signal to the next layer. The left chart of the video shows each neuron as a dot and each connection as a grey line. Each of these connections has a different weight (i.e. importance), represented by the thickness of the grey line. These weights are randomly initialised and modified throughout the training process.

The two features of the data (x and y axes of the right visual) are fed to the two input neurons at the left of the chart, and passed through the network – to reach the output neuron, at the right of the chart. This output neuron will output a probability of a data point to be of the class red or blue. Through the training process, the weight of each connections will be modified to minimise the classification error. Each red dotted connection indicates a weight being changed (learning process).

The right chart plots the blue and red dots on a chart, along with a contour plot of the predicted probabilities. Darker shades of red or blue indicate higher probability to belong to each class respectively. These predicted probabilities evolve over the training epochs – i.e. the number of training iterations.

## Implementation

These two visuals were built in Python, using a matplotlib figure of two subplots. The neural network trained was implemented using the scikit-learn MLPClassifier object. The neural network’s internal structure was built using custom built helper functions taking the weight matrix as input and outputting a list of edge and vertices to be plotted.

At each training iteration, the weight deltas were obtained by retrieving the weight matrix and comparing it with the previous iteration’s weight matrix. Connections with high weight deltas were dotted and coloured in red. The decision boundaries were also plotted at each iteration using a contour plot. Each frame was saved as an individual image. The individual images were then assembled into a video using the ffmpeg library. 





