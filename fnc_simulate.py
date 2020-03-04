import numpy as np

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

