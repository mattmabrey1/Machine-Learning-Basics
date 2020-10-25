#   Matthew Mabrey
#   Artificial Intelligence CSC-380
#   Dr. Yoon
#   Homework 9 Problem 3.2
#   Program Description: This program uses k-means clustering (k=6) to try to separate the input data in X_train
#   into 6 clusters and plots the data in a 3D graph with the ground truth, as well as plots when using 3 components
#   determined by Principle Component Analysis. It also prints the Adjusted Rand Index (ARI) performance and the
#   Normalized Mutual Information (NMI) performance.

#   We use sklearn packages as the foundation of this program to fit and score our values as well as
#   using a 3-D scatter plot courtesy of scikit-learn at
#   https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf)

X_train_file = open("X_train.txt","r")
X_final_test_file = open("X_test.txt","r")
y_train_file_file = open("y_train.txt","r")
y_final_test_file = open("y_test.txt","r")
activity_labels_file = open("activity_labels.txt","r")

X_train = np.loadtxt(X_train_file)
X_final_test = np.loadtxt(X_final_test_file)
y_train = np.loadtxt(y_train_file_file)
y_final_test = np.loadtxt(y_final_test_file)
activity_labels = np.loadtxt(activity_labels_file, dtype=str)

X_train_file.close()
X_final_test_file.close()
y_train_file_file.close()
y_final_test_file.close()
activity_labels_file.close()

k_means = cluster.KMeans(n_clusters=6)
km = k_means.fit(X_train)
labels = km.predict(X_final_test)


fig1 = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig1, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(X_final_test[:, 0], X_final_test[:, 40], X_final_test[:, 80],
               c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('tBodyAcc-XYZ')
ax.set_ylabel('tGravityAcc-XYZ')
ax.set_zlabel('tBodyAccJerk-XYZ')
ax.set_title("6 Clusters")
ax.dist = 10


fig2 = plt.figure(2, figsize=(10, 8))
ax = Axes3D(fig2, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(X_final_test[:, 0], X_final_test[:,40], X_final_test[:, 80], c=y_final_test, cmap="Paired", edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('tBodyAcc-XYZ')
ax.set_ylabel('tGravityAcc-XYZ')
ax.set_zlabel('tBodyAccJerk-XYZ')
ax.set_title('Ground Truth')
ax.dist = 10

pca = PCA(n_components=3)
proj = pca.fit_transform(X_final_test)

fig3 = plt.figure(3, figsize=(10, 8))
ax = Axes3D(fig3, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
               c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Principle Component 1')
ax.set_ylabel('Principle Component 2')
ax.set_zlabel('Principle Component 3')
ax.set_title("6 Clusters using PCA")
ax.dist = 10



fig4 = plt.figure(4, figsize=(10, 8))
ax = Axes3D(fig4, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=y_final_test, cmap="Paired", edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Principle Component 1')
ax.set_ylabel('Principle Component 2')
ax.set_zlabel('Principle Component 3')
ax.set_title('Ground Truth using PCA')
ax.dist = 10



print("\n The Adjusted Rand Index (ARI) performance for K-Means Clustering with 6 Clusters: ", metrics.adjusted_rand_score(y_final_test, labels), " \n")
print("\n The Normalized Mutual Information (NMI) performance for K-Means Clustering with 6 Clusters: ", metrics.normalized_mutual_info_score(y_final_test, labels), " \n")
print("Real activity label for the data\n", y_final_test)
print("Predicted K-Means Clustering with 6 Clusters label\n", labels)

plt.show()



