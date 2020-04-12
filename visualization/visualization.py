import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
from matplotlib import cm
from scipy.interpolate import griddata
import pandas as pd
from pandas.plotting import parallel_coordinates

# Global Arrays
p = []
q = []
g = []
a = []
p1 = []
q1 = []
g1 = []
group = []

# make arrays from csv
data = list(csv.reader(open('../data_orig.csv')))
for j in range(2, 6):
    group.append([])
    for i in range(0, 15):

      # Percent of each answer
      p.append(data[j][2+(3*i)])
      p.append(data[j][3+(3*i)])
      p.append(data[j][4+(3*i)])

      # Question #
      q.append(i+1)
      q.append(i+1)
      q.append(i+1)

      # Group #
      g.append(j-1)
      g.append(j-1)
      g.append(j-1)

      # Answer (0 = Yes, 1 = No, 2 = Undecided)
      a.append(0)
      a.append(1)
      a.append(2)

      # Percent, Question & Group for all Yesses
      p1.append(data[j][2+(3*i)])
      q1.append(i+1)
      g1.append(j-1)

      # Percent for all Noes
      group[j-2].append(data[j][3+(3*i)])

# Convert to np.array
p = np.asfarray(p, float)
q = np.asfarray(q, float)
g = np.asfarray(g, float)
a = np.asfarray(a, float)
p1 = np.asfarray(p1, float)
q1 = np.asfarray(q1, float)
g1 = np.asfarray(g1, float)
group = np.asfarray(group, float)

# Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title ('Responses for poll')
ax.set_xlabel('Questions')
ax.set_ylabel('Percentage')
ax.set_zlabel('Answers')
img = ax.scatter(q, p, a, c=g, cmap=plt.hot())
fig.colorbar(img)
plt.show()

# Bar Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title ('Percentage of Yesses')
ax.set_xlabel('Groups')
ax.set_ylabel('Questions')
ax.set_zlabel('Percentage')
img = ax.bar(q1, p1, g1, zdir='x')
plt.show()

# Parallel Coordinates
df = pd.DataFrame({'Group 1': group[0],
                  'Group 2': group[1],
                 'Group 3': group[2],
                'Group 4': group[3]
                  })

df = df.T
df['name'] = df.index

df.reset_index(drop=True)
plt.title ('Percentage of Noes')
parallel_coordinates(df, 'name')
plt.show()
