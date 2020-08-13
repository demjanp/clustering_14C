import numpy as np
import multiprocessing as mp
from scipy.stats import norm
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

from fnc_common import *
from fnc_dist import *
from fnc_sum import *
from fnc_simulate import *

def calc_clusters_hca(D, n):
	# cluster of dates into n clusters based on the distance matrix D using Hierarchical Cluster Analysis
	#
	# returns clusters = {label: [idx, ...], ...}; idx = index of date in the distance matrix D
	
	clusters_l = fcluster(linkage(calc_distances_pca(D), method = "ward", metric = "euclidean"), n, criterion = "maxclust")
	labels = np.unique(clusters_l)
	clusters = {}
	for label in labels:
		clusters[label] = np.where(clusters_l == label)[0].tolist()
	return clusters

def get_n_clusters_hca(dates, curve, clusters_n):
	# calculates clustering of dates using Hierarchical Cluster Analysis for n clusters
	#
	# returns clusters, means
	# 	clusters = {label: [idx, ...], ...}; idx = index in dates
	# 	means = {label: mean, ...}; mean = mean of the summed distributions of the calibrated dates within the cluster
	
	distributions = calibrate_multi(dates, curve)
	D = calc_distance_matrix(distributions)
	
	clusters = calc_clusters_hca(D, clusters_n)
	clusters = dict([(idx, clusters[ci]) for idx, ci in enumerate(list(clusters.keys()))])
	means = {}
	for ci in clusters:
		means[ci] = calc_mean_std(curve[:,0], sum_14c([distributions[idx] for idx in clusters[ci]]))[0]
	return clusters, means

def calc_silhouette(D, clusters):
	# calculate Silhouette of clustered dates as defined by Rousseeuw (1987, https://doi.org/10.1016/0377-0427(87)90125-7)
	# clusters = {label: [idx, ...], ...}; idx = index of date in the distance matrix D
	#
	# returns silhouette_score
	
	if len(clusters) < 2:
		return -1
	clusters_l = np.zeros(D.shape[0], dtype = int)
	for li, label in enumerate(list(clusters.keys())):
		clusters_l[clusters[label]] = li + 1
	return silhouette_score(D, clusters_l, metric = "precomputed")

def get_clusters_hca_worker(state_mp, counter_mp, pi, D_rnd_pool_mp, dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts, uniform):
	# worker process for randomization testing of clustering solutions
	
	while state_mp[0] > 0:
		if state_mp[0] == 1:
			continue
		if state_mp[0] == 2:
			dists = gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts, state_mp, counter_mp, pi, uniform)
			if dists is not None:
				D_rnd_pool_mp.append(calc_distance_matrix(dists))

def get_clusters_hca_master(state_mp, D_rnd_pool_mp, ps_mp, sils, npass, convergence):
	# master process for randomization testing of clustering solutions
	
	state_mp[0] = 2
	clu_max = max(list(sils.keys()))
	for clusters_n in sils:
		sils_rnd = []
		sils_prev = None
		c = 0
		todo = npass
		iter = -1
		while True:
			iter += 1
			while iter >= len(D_rnd_pool_mp):
				pass
			sils_rnd.append(calc_silhouette(D_rnd_pool_mp[iter], calc_clusters_hca(D_rnd_pool_mp[iter], clusters_n)))
			if len(sils_rnd) >= todo:
				sils_m = np.array(sils_rnd).mean()
				if sils_prev is not None:
					c = 1 - np.abs(sils_prev - sils_m) / sils_prev
				sils_prev = sils_m
				if c >= convergence:
					print("\nConverged at:", c)
					break
				todo *= 2
			print("\rClusters: %d/%d, Iteration: %d/%d, Conv: %0.4f         " % (clusters_n, clu_max, len(sils_rnd), max(todo, npass*2), c), end = "")
		sils_rnd = np.array(sils_rnd)
		s = sils_rnd.std()
		if s == 0:
			p = 0
		else:
			p = 1 - float(norm(sils_rnd.mean(), s).cdf(sils[clusters_n]))
		ps_mp[clusters_n] = p
	state_mp[0] = 0

def get_clusters_hca(dates, curve, npass = 1000, convergence = 0.99, uniform = False):
	# calculates clustering of dates using Hierarchical Cluster Analysis for different numbers of clusters
	# if uniform == True: assume uniform distribution of dates within clusters
	#
	# returns clusters, means, ps, sils
	# 	clusters = {n: {label: [idx, ...], ...}, ...}; n = number of clusters, idx = index in dates
	# 	means = {n: {label: mean, ...}, ...}; mean = mean of the summed distributions of the calibrated dates within the cluster
	#	ps = {n: p-value, ...}; p-value of the null hypothesis that the Silhouette for n clusters is the product of randomly distributed dates
	#	sils = {n: Silhouette, ...}
	
	distributions = calibrate_multi(dates, curve)
	D = calc_distance_matrix(distributions)
	dates = np.array([[c14age, uncert] for _, c14age, uncert in dates])
	curve_cal_age, curve_conv_age, curve_uncert = curve[:,0], curve[:,1], curve[:,2]
	
	uncerts = dates[:,1].astype(int)
	sum_obs = sum_14c(distributions)  # [[calBP, sum P], ...]
	if uniform:
		t1, t2 = calc_range(curve_cal_age, sum_obs, 0.9545)
		cal_bp_mean = (t1 + t2) / 2
		cal_bp_std = abs(t1 - t2) / 2
	else:
		cal_bp_mean, cal_bp_std = calc_mean_std(curve_cal_age, sum_obs)
	dates_n = dates.shape[0]
	
	n_cpus = N_CPUS
	manager = mp.Manager()
	state_mp = manager.list([1]) # 1: suspend, 2: run, 0: terminate
	counter_mp = manager.list([0]*(n_cpus - 1))
	D_rnd_pool_mp = manager.list()
	ps_mp = manager.dict() # {clusters_n: p, ...}
	
	clusters = {}  # {clusters_n: {ci: [idx, ...], ...}, ...}
	means = {}  # {clusters_n: {ci: mean, ...}, ...}
	sils = {} # {clusters_n: silhouette_score, ...}
	for clusters_n in range(2, len(dates) - 1):
		clusters_p = calc_clusters_hca(D, clusters_n)
		if len(clusters_p) != clusters_n:
			continue
		clusters[clusters_n] = dict([(idx, clusters_p[ci]) for idx, ci in enumerate(list(clusters_p.keys()))])
		sils[clusters_n] = calc_silhouette(D, clusters[clusters_n])
		means[clusters_n] = {}
		for ci in clusters[clusters_n]:
			means[clusters_n][ci] = calc_mean_std(curve[:,0], sum_14c([distributions[idx] for idx in clusters[clusters_n][ci]]))[0]
	
	procs = []
	for pi in range(n_cpus - 1):
		procs.append(mp.Process(target = get_clusters_hca_worker, args = (state_mp, counter_mp, pi, D_rnd_pool_mp, dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts, uniform)))
		procs[-1].start()
	procs.append(mp.Process(target = get_clusters_hca_master, args = (state_mp, D_rnd_pool_mp, ps_mp, sils, npass, convergence)))
	procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None

	return clusters, means, dict(ps_mp), sils

def get_opt_clusters(clusters, ps, sils, p_value):
	# find optimal number of clusters based on Silhouette scores and p-values of clustering solutions
	# clusters = {n: {label: [idx, ...], ...}, ...}; n = number of clusters, idx = index in dates
	# ps = {n: p-value, ...}; p-value of the null hypothesis that the Silhouette for n clusters is the product of randomly distributed dates
	# sils = {n: Silhouette score, ...}
	#
	# returns number of clusters
	
	clu_ns = np.array(sorted([n for n in clusters]), dtype = int)
	ps = np.array([ps[clu_n] for clu_n in clu_ns])
	sils = np.array([sils[clu_n] for clu_n in clu_ns])
	idxs = np.where(ps < p_value)[0]
	if not idxs.size:
		return None
	return clu_ns[idxs[np.argmax(sils[idxs])]]

