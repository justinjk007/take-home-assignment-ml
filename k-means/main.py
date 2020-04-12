import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# load the dataset from file
dataset = pd.read_csv('../data_mod.csv')
print("\n")

# change group labels to numbers
dataset.loc[dataset.group == 'G1', 'group'] = 0
dataset.loc[dataset.group == 'G2', 'group'] = 1
dataset.loc[dataset.group == 'G3', 'group'] = 2
dataset.loc[dataset.group == 'G4', 'group'] = 3

X = dataset[dataset.columns[1:4]].values
y = dataset.group.values

# y_pred contains the output such that each number represent the
# cluster number of input we gave in. So if one of the number is 0,
# the input X of that output belongs to cluster 0. We can use this as
# the color input to the scatter plot which will decide on the color
# based on the index. So if y is 0 it will chose yellow, 1 if green
# and so on automatically.
y_pred = KMeans(n_clusters=4, random_state=69).fit_predict(X)

# Plot the ground truth
fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# Here y_pred is the color input, so it decides what color each point
# should get. X[:,0] and so on simply slices the input by coloumns. So
# X[:,0] returns all the values belonging to coloumns under label Yes,
# X[:,1] returns all the values belonging to coloumns under label No
# and so on.
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Yes')
ax.set_ylabel('No')
ax.set_zlabel('Undecided')
ax.set_title('K-means clustering')
ax.dist = 12

fig.show()
plt.show()
