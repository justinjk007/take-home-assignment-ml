from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn import metrics
import pandas as pd
import numpy as np


def main():
    # load the dataset from file
    dataset = pd.read_csv('../data_mod.csv')
    print("\n")

    # change group labels to numbers
    dataset.loc[dataset.group == 'G1', 'group'] = 0
    dataset.loc[dataset.group == 'G2', 'group'] = 1
    dataset.loc[dataset.group == 'G3', 'group'] = 2
    dataset.loc[dataset.group == 'G4', 'group'] = 3

    # Split the dataset into test and train datasets, 30% of datasets for testing
    train_X, test_X, train_Y, test_Y = train_test_split(
        dataset[dataset.columns[1:4]].values,  # input values aka the x values
        dataset.group.values,  # output values aka the y values
        test_size=0.3,  # move 30% into test data
        random_state=0,
    )
    # convert output data to lists
    train_Y = train_Y.astype(np.float32)
    test_Y = test_Y.astype(np.float32)

    nb = BernoulliNB()

    nb_fit = nb.fit(train_X, train_Y)
    output = nb_fit.predict(test_X)
    print("Predicted output: ", output)
    print("Expected output : ", test_Y)
    match = 0
    for i, j in zip(output, test_Y):
        if (i == j):
            match += 1
    print("Number of matches out of a total %d cases : %d" %
          (test_X.shape[0], match))


if __name__ == "__main__":
    main()
