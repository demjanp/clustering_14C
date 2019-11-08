import numpy as np

def sum_14c(distributions):
	# returns sum of distributions normalized to 1
	
	summed = np.zeros(distributions[0].shape[0])
	for distr in distributions:
		summed += distr
	s = summed.sum()
	if s > 0:
		summed /= s
	return summed

def calc_mean_std(values, weights):
	# returns weighted mean and standard deviation of values
	
	mean = (values * weights).sum()
	std = np.sqrt((((values - mean)**2) * weights).sum())
	return mean, std

