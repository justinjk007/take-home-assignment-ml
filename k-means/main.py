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

y_pred = KMeans(n_clusters=4, random_state=69).fit_predict(X)

# Plot the ground truth
fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

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
