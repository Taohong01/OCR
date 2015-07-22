"""
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
print 'X:\n', X
print X.shape
pca = PCA(n_components=2)
pca.fit(X)
print 'get precision', pca.get_precision()

print
print
print 'variance ratio\n'
print(pca.explained_variance_ratio_) 
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(np.array([2,1.5]))
print indices                                           
print distances
help(nbrs)