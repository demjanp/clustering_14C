import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA

def p_same_event(dist1, dist2):
	# returns probability that distributions dist1 and dist2 represent the same event
	
	return (4*(dist1 * dist2).sum()) / (dist1.sum() + dist2.sum())**2

def calc_distance_matrix(distributions):
	# returns a distance matrix of distributions based on probabilities that they represent the same events
	
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

def calc_distance_matrix_worker(params):
	
	dist1, dist2 = params
	
	return p_same_event(dist1, dist2)

def calc_distance_matrix_mp(distributions, pool):
	# multiprocessing version of calc_distance_matrix
	
	dists_n = len(distributions)
	PS = np.zeros((dists_n, dists_n), dtype = float)
	ds = list([d1, d2] for d1, d2 in combinations(range(dists_n), 2))
	ps_diag = pool.map(calc_distance_matrix_worker, ([distributions[d1], distributions[d2]] for d1, d2 in ds))
	for idx in range(len(ds)):
		PS[ds[idx][0], ds[idx][1]] = ps_diag[idx]
		PS[ds[idx][1], ds[idx][0]] = ps_diag[idx]
	D = 1 - PS
	D = D - D.min()
	for i in range(D.shape[0]):
		D[i, i] = 0
	return D

def calc_distances_pca(distances, n_components = None):
	# returns Principal Component Analysis scores for each date based on their distances as defined in calc_distance_matrix
	
	if n_components is None:
		pca = PCA(n_components = None)
		pca.fit(distances)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0]
		if not n_components.size:
			n_components = 1
		else:
			n_components = n_components.min() + 1
	pca = PCA(n_components = n_components)
	pca.fit(distances)
	return pca.transform(distances)

