import numpy as np
import multiprocessing as mp

from fnc_common import *
from fnc_sum import *

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

def get_randomized_worker(params):
	
	dates_n, cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max = params
	return gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max)

def get_randomized(dates, curve, pool, p_diff_max = 0.001):
	
	distributions = calibrate_multi(dates, curve)
	dates = np.array([[c14age, uncert] for _, c14age, uncert in dates])
	curve_cal_age, curve_conv_age, curve_uncert = curve[:,0], curve[:,1], curve[:,2]
	
	uncerts = dates[:,1].astype(int)
	sum_obs = sum_14c(distributions)  # [[calBP, sum P], ...]
	cal_bp_mean, cal_bp_std = calc_mean_std(curve_cal_age, sum_obs)
	idxs = np.where(sum_obs > 0)[0]
	idx_min, idx_max = int(idxs.min()), int(idxs.max())
	
	dist_rnd_pool = []
	
	step = 100
	iters = step
	collect_dist = []
	sums = []
	sums_mean_last = None
	iter = -1
	while len(collect_dist) < iters:
		iter += 1
		if iter >= len(dist_rnd_pool):
			dist_rnd_pool += pool.map(get_randomized_worker, ([len(dates), cal_bp_mean, cal_bp_std, curve, curve_cal_age, curve_conv_age, curve_uncert, uncerts, idx_min, idx_max] for i in range(mp.cpu_count())))
		collect_dist.append(dist_rnd_pool[iter])
		sums.append(sum_14c(dist_rnd_pool[iter]))
		sums_mean = np.mean(sums, axis = 0)
		if sums_mean_last is not None:
			s_diff = np.abs(sums_mean - sums_mean_last).sum()
			if s_diff >= p_diff_max:
				iters = len(collect_dist) + step
			print("\rgen. rnd. %d/%d diff: %0.4f     " % (len(collect_dist), iters, s_diff), end = "")  # DEBUG
		sums_mean_last = sums_mean
	
	return curve_cal_age[idx_min:idx_max], [dist[idx_min:idx_max] for dist in distributions], sum_obs[idx_min:idx_max], collect_dist, sums

