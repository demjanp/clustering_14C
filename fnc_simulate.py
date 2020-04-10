import numpy as np
import multiprocessing as mp
from scipy.stats import norm
from scipy.optimize import basinhopping

from fnc_common import *
from fnc_sum import *

def gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts, state_mp, counter_mp, pi):
	# generate randomized distributions based on observed 14C dates
	# dates_n = number of generated dates (distributions)
	# cal_bp_mean, cal_bp_std = Mean and Standard Deviation of the sum of the generated distributions
	# curve_cal_age, curve_conv_age, curve_uncert = first three columns columns from the calibration curve
	# uncerts = [14C years, ...]; uncertainties of 14C dates to choose from
	# state_mp, counter_mp, pi = multiprocessing helper variables
	#
	# returns distributions = [distribution, ...]; distribution = [p, ...]; p in order of curve_cal_age
	
	def get_dists(ages_14c, uncerts_rnd):
		# calculate distributions based on 14C ages and uncertainties
		
		return [calibrate(age_14c, uncert, curve_conv_age, curve_uncert) for age_14c, uncert in zip(ages_14c, uncerts_rnd)]
	
	def fnc_fitness(ages_14c):
		# function to be optimized
		#
		# returns summed squared distance of mean and standard deviation of summed distribution of ages_14c from cal_bp_mean and cal_bp_std
		
		if state_mp[0] != 2:
			return np.inf
		if counter_mp[pi] > 1000:
			return np.inf
		counter_mp[pi] += 1
		sum_rnd = sum_14c(get_dists(ages_14c, uncerts_rnd))
		idxs = np.where(sum_rnd > 0)[0]
		idx0, idx1 = int(idxs.min()), int(idxs.max())
		mean_rnd, std_rnd = calc_mean_std(curve_cal_age[idx0:idx1], sum_rnd[idx0:idx1])
		if max(abs(mean_rnd - cal_bp_mean), abs(std_rnd - cal_bp_std)) < 1:
			return 0
		return (mean_rnd - cal_bp_mean)**2 + (std_rnd - cal_bp_std)**2
	
	while True:
		if state_mp[0] != 2:
			return None
		uncerts_rnd = [np.random.choice(uncerts) for i in range(dates_n)]
		# generate initial guess by simulating 14C dates of samples with normally distributed calendar ages with cal_bp_mean, cal_bp_std
		ages_14c = [curve_conv_age[np.argmin(np.abs(curve_cal_age - np.random.normal(cal_bp_mean, cal_bp_std)))] for i in range(dates_n)]
		counter_mp[pi] = 0
		# use basin-hopping algorithm to minimize distance of mean and standard deviation of summed distribution of ages_14c from cal_bp_mean and cal_bp_std
		res = basinhopping(fnc_fitness, ages_14c, niter_success = 1)
		if res.lowest_optimization_result.fun == 0:
			break
	return get_dists(res.x, uncerts_rnd)

def get_randomized_worker(state_mp, counter_mp, pi, distributions_mp, dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts):
	# worker process to generate randomized distributions
	
	while state_mp[0] > 0:
		if state_mp[0] == 1:
			continue
		if state_mp[0] == 2:
			distributions_mp.append(gen_random_dists(dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts, state_mp, counter_mp, pi))

def get_randomized_master(state_mp, distributions_mp, sums_mp, npass, convergence):
	# master process to generate sets of randomized distributions
	
	state_mp[0] = 2
	sums = []
	sums_prev = None
	c = 0
	todo = npass
	while True:
		if len(distributions_mp) >= todo:
			sums += [sum_14c(dist) for dist in distributions_mp[-(len(distributions_mp) - len(sums)):] if (dist is not None)]
			sums_m = np.array(sums).mean(axis = 0)
			if sums_prev is not None:
				c = ((sums_prev * sums_m).sum()**2) / ((sums_prev**2).sum() * (sums_m**2).sum())
			sums_prev = sums_m.copy()
			if c >= convergence:
				print("\nConverged at:", c)
				break
			todo *= 2
		print("\rRandomizing distributions, Iteration: %d/%d, Conv: %0.4f         " % (len(distributions_mp), max(todo, npass*2), c), end = "")
	state_mp[0] = 0
	sums_mp += sums

def get_randomized(dates, curve, npass = 100, convergence = 0.99):
	# generate sets of randomized distributions based on observed 14C dates
	# dates = observed 14C dates; [[lab_code, c14age, uncert], ...]
	# curve = calibration curve; [[calendar age, 14C age, uncertainty], ...]
	# npass = initial number of passes before calculating convergence (doubles after every calculation)
	# convergence = minimum required level of convergence to stop iterations
	#
	# returns curve_cal_age, distributions, summed, distributions_rnd, summed_rnd, p
	# 	curve_cal_age = [calendar years, ...]; first column of the calibration curve
	# 	distributions = distributions of calibrated observed 14C dates; [distribution, ...]; distribution = [p, ...]; p in order of curve_cal_age
	# 	summed = summed distributions; [sum p, ...]; p in order of curve_cal_age
	# 	distributions_rnd = sets of randomized distributions; [distributions, ...]
	#	summed_rnd = summed sets of randomized distributions; [summed, ...]; in order of distributions_rnd
	# 	p = probability of randomly producing distributions whose sum is at least as extreme as the sum of the observed distributions (p-value) if the sum of the randomly generated distributions has the same mean and standard deviation as the sum of the observed ones
	
	# calibrate observed 14C dates
	distributions = calibrate_multi(dates, curve)
	dates = np.array([[c14age, uncert] for _, c14age, uncert in dates])
	curve_cal_age, curve_conv_age, curve_uncert = curve[:,0], curve[:,1], curve[:,2]
	
	# extract uncertainties of observed 14C dates
	uncerts = dates[:,1].astype(int)
	# calculate mean and standard deviation of the summed distributions of the calibrated observed 14C dates
	sum_obs = sum_14c(distributions)  # [[calBP, sum P], ...]
	cal_bp_mean, cal_bp_std = calc_mean_std(curve_cal_age, sum_obs)
	dates_n = dates.shape[0]
	
	# generate sets of randomized distributions and their sums
	n_cpus = mp.cpu_count()
	manager = mp.Manager()
	state_mp = manager.list([1]) # 1: suspend, 2: run, 0: terminate
	counter_mp = manager.list([0]*(n_cpus - 1))
	distributions_mp = manager.list()
	sums_mp = manager.list()
	procs = []
	for pi in range(n_cpus - 1):
		procs.append(mp.Process(target = get_randomized_worker, args = (state_mp, counter_mp, pi, distributions_mp, dates_n, cal_bp_mean, cal_bp_std, curve_cal_age, curve_conv_age, curve_uncert, uncerts)))
		procs[-1].start()
	procs.append(mp.Process(target = get_randomized_master, args = (state_mp, distributions_mp, sums_mp, npass, convergence)))
	procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	sums_rnd = np.array(sums_mp)
	
	# calculate p-value
	mask = (sums_rnd.std(axis = 0) > 0)
	p = (1 - norm(sums_rnd[:,mask].mean(axis = 0), sums_rnd[:,mask].std(axis = 0)).cdf(sum_obs[mask])).min()
	
	distributions_mp = [dist for dist in distributions_mp if (dist is not None)]
	
	return curve_cal_age, distributions, sum_obs, list(distributions_mp), sums_rnd, p

