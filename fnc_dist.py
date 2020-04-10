import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA

def p_same_event(dist1, dist2):
	# probability that distributions dist1 and dist2 represent the same event
	#
	# returns probability value
	
	return (4*(dist1 * dist2).sum()) / (dist1.sum() + dist2.sum())**2

def calc_distance_matrix(distributions):
	# calculate a distance matrix of distributions of calibrated 14C dates based on probabilities that they represent the same event
	# distributions = [distribution, ...]; distribution = [p, ...]
	#
	# returns D[i,j] = d; i,j = index in distributions; d = inverse probability that distributions i and j represent the same event
	
	dists_n = len(distributions)
	PS = np.zeros((dists_n, dists_n), dtype = float)
	for d1, d2 in combinations(range(dists_n), 2):
		p = p_same_event(distributions[d1], distributions[d2])
		PS[d1,d2] = p
		PS[d2,d1] = p
	D = 1 - PS
	D = D - D.min()
	for i in range(D.shape[0]):
		D[i, i] = 0
	return D

def calc_distances_pca(D, n_components = None):
	# Principal Component Analysis scores for each date based on their distances as defined in calc_distance_matrix
	# number of components is chosen so that they explain 99% of variance
	# D[i,j] = d; i,j = index in distributions; d = inverse probability that distributions i and j represent the same event
	#
	# returns S[i,k] = PCA score; i = index in distributions; k = index of PCA component
	
	if n_components is None:
		pca = PCA(n_components = None)
		pca.fit(D)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0]
		if not n_components.size:
			n_components = 1
		else:
			n_components = n_components.min() + 1
	pca = PCA(n_components = n_components)
	pca.fit(D)
	return pca.transform(D)

