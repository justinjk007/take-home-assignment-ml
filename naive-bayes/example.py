#######################################################################
# Example demonstrating iris flower classification with 100% accuracy #
#######################################################################

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

gnb = GaussianNB()
nb_fit = gnb.fit(X_train, y_train)
output = nb_fit.predict(X_test)
print("Predicted output: ", output)
print("Expected output : ", y_test)
match = 0
for i, j in zip(output, y_test):
    if (i == j):
        match += 1

print("Number of matches out of a total %d cases : %d" %
      (X_test.shape[0], match))
