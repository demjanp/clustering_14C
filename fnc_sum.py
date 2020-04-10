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

def calc_percentiles(distributions, perc_lower, perc_upper):
	# calculate lower and upper percentile of a set of probability distributions
	# distributions = [distribution, ...]; distribution = [p, ...]
	# returns dist_lower, dist_upper = [p, ...]
	
	size = len(distributions[0]) * len(distributions)
	if size > 50000000:
		dist_lower = []
		dist_upper = []
		for i in range(len(distributions[0])):
			column = np.array([row[i] for row in distributions])
			dist_lower.append(np.percentile(column, perc_lower))
			dist_upper.append(np.percentile(column, perc_upper))
		dist_lower = np.array(dist_lower)
		dist_upper = np.array(dist_upper)
	else:
		dist_lower = np.percentile(distributions, perc_lower, axis = 0)
		dist_upper = np.percentile(distributions, perc_upper, axis = 0)
	
	return dist_lower, dist_upper
