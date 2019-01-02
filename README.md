# Pokerhand-Predictor

Predicts the class of a poker hand given the cards and suits

The training.py file allows for training of a tensorflow model.
The training set is the poker-hand-training.txt and the testing set is poker-hand-testing.txt.
Information on the data in the sets is in poker-hand.names.txt, this dataset is taken from
[UCI Machine Learning Dataset](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)
Current model is a model with 2 hidden layers:

+ First layer has 256 nodes and uses the rectified linear function,
+ Second layer has 256 nodes and uses the sigmoid function

Output layer is softmax with 10 values.
Training uses 20% of the training dataset for validation.

baseline.py generates the baseline model which is saved in SavedModels.
