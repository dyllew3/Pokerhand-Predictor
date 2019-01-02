import csv
import tensorflow as tf
import numpy as np

from tensorflow import keras


mapping = {

    1: [0.0, 0.0, 1.0],
    2: [0.0, 1.0, 0.0],
    3: [1.0, 0.0, 0.0],
    4: [0.0, 0.0, 0.0]
}


def get_data(filename, delimiter=','):
    all_data = np.genfromtxt(filename, delimiter=delimiter)
    return all_data

def new_rows(filename):
    result = []
    for row in get_data(filename):
        temp_res = []
        for i in range(len(row)):
            new_element = [row[i]] if i % 2 == 1 or i==10 else  mapping[row[i]]
            temp_res += (new_element)
        result.append(temp_res)
    return result

def main():
    a  = (new_rows('Data/poker-hand-training.txt'))
    with open('Data/poker-hand-training-converted.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(a)
        f.close()
    b  = (new_rows('Data/poker-hand-testing.txt'))
    with open('Data/poker-hand-testing-converted.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(b)
        f.close()
    pass

if __name__ == "__main__":
    main()    
