# UIUC CS 440 - Artifical Intelligence
My work for [CS440 AI](https://cs.illinois.edu/academics/courses/cs440). Topics include search, recognition, neural nets, and reinforcement learning

## MP1: Search
Implement various types of Search to find waypoints in a maze. Includes:
- Single waypoint (start to end)
- 4 waypoint (corners)
- Multi-waypoint (arbitrary number & locations)

The best algorithm implemented here was an A* search using a Minimum Spanning Tree as a heuristic.

## MP2: Naive Bayes
Implemented a Naive Bayes classifier which, given sufficient training data, is able to distinguish between Spam emails and Non-spam emails.

Uses both a single-word (unigram) and two-word pair (bigram) model, as well as a mixed model. Achieves 80-90% accuracy depending on test dataset.

## MP3: Neural Networks
Implemented shallow network (1 hidden layer) as well as a deeper Convolutional neural network to classify images in the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image dataset.

## MP4: Hidden Markov Models
Implemented the Viterbi algorithm on a  to tag words in a sentence with part of speech. 

Achieved over 95% accuracy on test dataset.

## MP5: Alphabeta search (Chess)
Implemented Minimax and Alphabeta search to play chess optimally. Also implemented a stochastic search, which is faster but less optimal.

# MP6: Reinforcement learning
Implemented Model-free reinforcement learning to (virtually) balance a pole atop a movable cart.
