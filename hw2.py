from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import mode
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0);
# Get best(top) k
k_finder = KElbowVisualizer(KMeans(), k=(1,20))
k_finder.fit(X)

top_k = k_finder.elbow_value_

# Do kmenas and calc accuracy
k_means = KMeans(n_clusters=top_k).fit(X)
y_kmeans = k_means.fit_predict(X)
labels = np.zeros_like(y_kmeans)

for k in range(top_k):
	mask = (y_kmeans == k)
	labels[mask] = mode(y_true[mask], keepdims = False)[0]

score = accuracy_score(y_true, labels)
print("Accuracy: " + repr(score * 100) + " %")

# Confusion matrix
c_mat = confusion_matrix(y_true, labels);
sns.heatmap(c_mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.show();
