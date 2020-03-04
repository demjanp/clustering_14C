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

def calc_percentiles(ps, perc_lower, perc_upper):
	
	size = len(ps[0]) * len(ps)
	if size > 50000000:
		p_lower = []
		p_upper = []
		for i in range(len(ps[0])):
			column = np.array([row[i] for row in ps])
			p_lower.append(np.percentile(column, 5))
			p_upper.append(np.percentile(column, 95))
		p_lower = np.array(p_lower)
		p_upper = np.array(p_upper)
	else:
		p_lower = np.percentile(ps, perc_lower, axis = 0)
		p_upper = np.percentile(ps, perc_upper, axis = 0)
	
	return p_lower, p_upper
