import numpy as np
import multiprocessing as mp
from scipy.stats import norm
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

from fnc_common import *
from fnc_dist import *
from fnc_sum import *

def calc_clusters_hca(D, n):
	# returns clustering of dates into n clusters based on the distance matrix D using Hierarchical Cluster Analysis
	
	clusters_l = fcluster(linkage(calc_distances_pca(D), method = "ward", metric = "euclidean"), n, criterion = "maxclust")
	labels = np.unique(clusters_l)
	clusters = {}
	for label in labels:
		clusters[label] = np.where(clusters_l == label)[0].tolist()
	return clusters

def calc_silhouette(D, clusters):
	# returns Silhouette of clustered dates as defined by Rousseeuw (1987, https://doi.org/10.1016/0377-0427(87)90125-7)
	# clusters = {label: [idx, ...], ...}; idx = index of date in the distance matrix D
	
	if len(clusters) < 2:
		return -1
	clusters_l = np.zeros(D.shape[0], dtype = int)
	for li, label in enumerate(list(clusters.keys())):
		clusters_l[clusters[label]] = li + 1
	return silhouette_score(D, clusters_l, metric = "precomputed")

def gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max):
	# returns randomly generated distributions
	# dates_n = number of generated dates (distributions)
	# cal_bp_mean, cal_bp_std = Mean and Standard Deviation of the sum of the generated distributions
	# curve = calibration curve
	# curve_cal_age, curve_conv_age, curve_uncert = first three columns columns from the calibration curve
	# idx_min, idx_max = minimal and maximal index delimiting non-zero values in distributions (used to restrict calculations only on the relevant areas of the calibration curve)
	
	def _make_random_dist(t_mean, t_dev):
		
		t_dev = max(1, min(curve_cal_age[idx_max] - t_mean, t_mean - curve_cal_age[idx_min], t_dev))
		while True:
			cal_age_bp = np.random.normal(t_mean, t_dev)
			c14_age, error = curve[np.argmin(np.abs(curve_conv_age - cal_age_bp))][1:3]
			c14_age = np.random.normal(c14_age, error, 1)
			uncert = int(np.random.choice(uncerts))
			dist = calibrate(c14_age, uncert, curve_conv_age, curve_uncert)
			dist = (dist / dist.sum())[idx_min:idx_max]
			if dist.sum() == 1:
				return dist
	
	def _calc_diff(distributions):
		
		m, s = calc_mean_std(curve_cal_age[idx_min:idx_max], sum_14c(distributions))
		return max(abs(cal_bp_mean - m), abs(cal_bp_std - s))
	
	t_dev = (curve_cal_age[idx_max] - curve_cal_age[idx_min]) / 2
	distributions_rnd = [_make_random_dist(cal_bp_mean, t_dev) for i in range(dates_n)]
	diff = _calc_diff(distributions_rnd)
	diff_opt = diff
	while True:
		diffs = np.array([_calc_diff(distributions_rnd[:idx] + distributions_rnd[idx + 1:]) for idx in range(0, dates_n)])
		idx = np.argmin(diffs)
		m = calc_mean_std(curve_cal_age[idx_min:idx_max], sum_14c(distributions_rnd))[0]
		iters = 0
		while True:
			distributions_rnd[idx] = _make_random_dist(cal_bp_mean + (cal_bp_mean - m) / 2, t_dev / 2)
			diff = _calc_diff(distributions_rnd)
			if diff < diff_opt:
				diff_opt = diff
				iters = 0
				break
			iters += 1
			if iters >= dates_n**2:
				distributions_rnd = [_make_random_dist(cal_bp_mean, t_dev) for i in range(dates_n)]
				diff_opt = _calc_diff(distributions_rnd)
				break
		if diff_opt < 1:
			break
	return distributions_rnd

def get_clusters_hca_worker(params):
	
	dates_n, cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max = params
	return calc_distance_matrix(gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max))

def get_clusters_hca(dates, curve, pool, p_diff_max = 0.001):
	# calculates clustering of dates using Hierarchical Cluster Analysis for different numbers of clusters
	# p_diff_max = maximum change in p-value when generating randomized datasets (lower = more stable solution)
	# returns clusters, means, ps, sils
	# 	clusters = {n: {label: [idx, ...], ...}, ...}; n = number of clusters, idx = index in dates
	# 	means = {n: {label: mean, ...}, ...}; mean = mean of the summed distributions of the calibrated dates within the cluster
	#	ps = {n: p-value, ...}; p-value of the null hypothesis that the Silhouette for n clusters is the product of randomly distributed dates
	#	sils = {n: Silhouette, ...}
	
	distributions = calibrate_multi(dates, curve)
	D = calc_distance_matrix_mp(distributions, pool)
	dates = np.array([[c14age, uncert] for _, c14age, uncert in dates])
	curve_cal_age, curve_conv_age, curve_uncert = curve[:,0], curve[:,1], curve[:,2]
	
	uncerts = dates[:,1].astype(int)
	sum_obs = sum_14c(distributions)  # [[calBP, sum P], ...]
	cal_bp_mean, cal_bp_std = calc_mean_std(curve_cal_age, sum_obs)
	idxs = np.where(sum_obs > 0)[0]
	idx_min, idx_max = int(idxs.min()), int(idxs.max())
	
	clusters = {}  # {clusters_n: {ci: [idx, ...], ...}, ...}
	means = {}  # {clusters_n: {ci: mean, ...}, ...}
	ps = {} # {clusters_n: p, ...}
	sils = {} # {clusters_n: silhouette_score, ...}
	vals = np.linspace(D.min(), D.max(), max(100, len(dates) * 2))
	
	D_rnd_pool = []
	
	clu_max = len(dates) - 2
	for clusters_n in range(2, clu_max + 1):
		
		clusters_p = calc_clusters_hca(D, clusters_n)
		
		if len(clusters_p) != clusters_n:
			continue
		
		clusters_p = dict([(idx, clusters_p[ci]) for idx, ci in enumerate(list(clusters_p.keys()))])
		
		sils[clusters_n] = calc_silhouette(D, clusters_p)
		
		sils_rnd = None
		p_last = None
		step = 100
		iters = step
		iter = -1
		while True:
			iter += 1
			if iter >= len(D_rnd_pool):
				D_rnd_pool += pool.map(get_clusters_hca_worker, ([len(dates), cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max] for i in range(mp.cpu_count())))
			clusters_rnd = calc_clusters_hca(D_rnd_pool[iter], clusters_n)
			sil_rnd = calc_silhouette(D_rnd_pool[iter], clusters_rnd)
			if sils_rnd is None:
				sils_rnd = np.array([sil_rnd])
			else:
				sils_rnd = np.hstack((sils_rnd, [sil_rnd]))
			s = sils_rnd.std()
			if s == 0:
				p = 0
			else:
				p = 1 - float(norm(sils_rnd.mean(), s).cdf(sils[clusters_n]))
			print("\rClusters: %d/%d, Iteration: %d/%d        " % (clusters_n, clu_max, len(sils_rnd), iters), end = "")
			if p_last is not None:
				if abs(p - p_last) < p_diff_max:
					if sils_rnd.shape[0] >= iters:
						break
				else:
					iters = sils_rnd.shape[0] + step
			p_last = p
		
		ps[clusters_n] = p
		
		clusters[clusters_n] = clusters_p
		
		means[clusters_n] = {}
		for ci in clusters_p:
			means[clusters_n][ci] = calc_mean_std(curve[:,0], sum_14c([distributions[idx] for idx in clusters_p[ci]]))[0]
	
	return clusters, means, ps, sils

def get_opt_clusters(clusters, ps, sils, p_value):
	
	clu_ns = np.array(sorted([n for n in clusters]), dtype = int)
	ps = np.array([ps[clu_n] for clu_n in clu_ns])
	sils = np.array([sils[clu_n] for clu_n in clu_ns])
	idxs = np.where(ps < p_value)[0]
	if not idxs.size:
		return None
	return clu_ns[idxs[np.argmax(sils[idxs])]]

