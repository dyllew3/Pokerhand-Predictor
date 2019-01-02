import numpy as np
import tensorflow as tf
import numpy as np

from tensorflow import keras


# Reads csv file and converts it to numpy array
# Returns a tuple of features and outputs
def get_data(filename, tests, amount=0,  valid=0, delimiter=',',):
    all_data = np.genfromtxt(filename, delimiter=delimiter)
    if not tests:
        result = np.random.choice(range(all_data.shape[0]), amount + valid,replace=False)
        all_data = np.array([all_data[x] for x in result])
    # Numpy array of all feature arrays
    # Each feature array has 5 features
    features = all_data[:, :-1]
    # Numpy array of all the outputs(y values)
    labels = all_data[:,all_data.shape[1] - 1].ravel()

    return features, labels

test_data, test_labels = get_data('Data/poker-hand-testing.txt', True)
test_labels = keras.utils.to_categorical(test_labels, 10)

def train_model(train_data, train_labels, amount, optimizer=tf.train.AdamOptimizer(), epochs=300, batch_size=512):
    valid = amount // 4
    model = keras.Sequential()

    # One layer with rectified linear and 256 nodes
    model.add(keras.layers.Dense(256, activation=tf.nn.relu,input_shape=(10,)))
    #model.add(keras.layers.Dropout(0.5))

    # Second layer has 256 nodes and uses the sigmoid function
    model.add(keras.layers.Dense(256, activation=tf.nn.sigmoid))
    #model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    # Training algorithm
    model.compile(optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])
        # Not necessary but helpful to see what's going on in the model
    #model.summary()


    # Epochs are just number of iterations
    # Batch size is just the data set divided up into mini-batches of 512 samples
    # Recommend playing around with epochs and batch size
    model.fit(train_data[valid:],
                train_labels[valid:],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(train_data[:valid], train_labels[:valid]),
                verbose=0)
    return model

def get_average(amount,optimizer, num_splits=1):
    valid = amount // 4
    result = []

    for _ in range(4):
        train_data, train_labels = get_data('Data/poker-hand-training.txt', False, amount, valid )
        train_labels = keras.utils.to_categorical(train_labels, 10)

        model = train_model(train_data, train_labels, amount)
        
        # Gets accuracy of model trained with training data against test data
        # result is an array with loss as first element and accuracy as 2nd element
        result.append(model.evaluate(test_data, test_labels))
    return np.mean(np.array(result), axis=0)

def run_tests(optimizer):
    amounts =[20000]
    averages = []
    for i in amounts:
        averages.append(get_average(i,optimizer))
    for i in range(len(averages)):
        print("Average of %i" % amounts[i])
        print(averages[i][::-1])
        print("----------------")
    pass

def main():
    run_tests(tf.train.AdamOptimizer())

if __name__ == "__main__":
    main()
