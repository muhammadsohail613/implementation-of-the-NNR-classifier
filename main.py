import numpy as np
import time
import json
import pandas as pd
import random

from sklearn.metrics import accuracy_score
from typing import List


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    # Find optimal radius
    best_radius = None
    best_accuracy = 0

    data_trn=pd.read_csv(data_trn)
    data_vld=pd.read_csv(data_vld)
    data_tst=pd.read_csv(data_tst)

    labels_trn = data_trn['class'].values
    labels_vld = data_vld['class'].values
    labels_tst = data_tst['class'].values

    data_trn = data_trn.drop(['class'], axis=1)
    data_vld = data_vld.drop(['class'], axis=1)
    data_tst = data_tst.drop(['class'], axis=1)

    # Converting all to numertical in case they are not, if a categorical 
    # column occurs it adds NAN to it 
    data_trn = data_trn.apply(pd.to_numeric, errors='coerce')
    data_vld = data_vld.apply(pd.to_numeric, errors='coerce')
    data_tst = data_tst.apply(pd.to_numeric, errors='coerce')

    # Tested that 21 is the best radius for accuracy
    for radius in range(1, 21): # you can change the range of radius
        accuracy = 0
        for i in range(data_vld.shape[0]):
            test_instance = data_vld.iloc[i, :].values
            distances = np.linalg.norm(data_trn - test_instance, axis=1)
            neighbors = data_trn[distances <= radius]
            if len(neighbors) > 0:
                majority_class = max(labels_trn[distances <= radius], key=list(labels_trn[distances <= radius]).count)
                if majority_class == labels_vld[i]:
                    accuracy += 1

        accuracy = accuracy / data_vld.shape[0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius

    print("Optimal Radius: ", best_radius)

    # Classify instances in test set
    accuracy = 0
    predictions = []
    for i in range(data_tst.shape[0]):
        test_instance = data_tst.iloc[i, :].values
        distances = np.linalg.norm(data_trn - test_instance, axis=1)
        neighbors = data_trn[distances <= best_radius]
        if len(neighbors) > 0:
            majority_class = max(labels_trn[distances <= best_radius], key=list(labels_trn[distances <= best_radius]).count)
            predictions.append(majority_class)
            if majority_class == labels_tst[i]:
                accuracy += 1
        else:
            majority_class = random.choice(labels_trn)
            predictions.append(majority_class)
    accuracy = accuracy / data_tst.shape[0]
    print("Accuracy on test set: ", accuracy)
    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
