import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d

N_CPUS = min(30, mp.cpu_count() - 1)

def load_calibration_curve(fcalib, interpolate = False):
	# load calibration curve
	# data from: fcalib 14c file
	# returns: [[CalBP, ConvBP, CalSigma], ...], sorted by CalBP
	
	with open(fcalib, "r", encoding="latin1") as f:
		data = f.read()
	data = data.split("\n")
	cal_curve = []
	for line in data:
		line = line.strip()
		if not line:
			continue
		if line.startswith("#"):
			continue
		cal_curve.append([np.float64(value) for value in line.split(",")])
	cal_curve = np.array(cal_curve, dtype = np.float64)
	cal_curve = cal_curve[np.argsort(cal_curve[:,0])]
	
	if interpolate:
		cal_bp = np.arange(cal_curve[:,0].min(), cal_curve[:,0].max() + 1)
		cal_curve = np.vstack((
			cal_bp,
			interp1d(cal_curve[:,0], cal_curve[:,1], kind = "quadratic")(cal_bp),
			interp1d(cal_curve[:,0], cal_curve[:,2], kind = "linear")(cal_bp),
		)).T
	
	return cal_curve.astype(np.float64)

def load_dates(fname):
	# load dates from text file where each row represents a date in the format: "[Lab Code], [14C Age], [Uncertainty]"
	#
	# returns [[lab_code, c14age, uncert], ...]
	
	with open(fname, "r", encoding="latin1") as f:
		rows = f.read()
	dates = []
	rows = rows.split("\n")
	for row in rows:
		row = row.strip()
		if row:
			row = row.split(",")
			lab_code, c14age, uncert  = [value.strip() for value in row]
			c14age, uncert = float(c14age), float(uncert)
			dates.append([lab_code, c14age, uncert])
	return dates

def calibrate(age, uncert, curve_conv_age, curve_uncert):
	# calibrate a 14C date
	# calibration formula as defined by Bronk Ramsey 2008, doi: 10.1111/j.1475-4754.2008.00394.x
	# age: uncalibrated 14C age BP
	# uncert: 1 sigma uncertainty
	#
	# returns distribution = [p, ...]; p = probability value of the corresponding calendar age on the calibration curve
	
	sigma_sum = uncert**2 + curve_uncert**2
	return (np.exp(-(age - curve_conv_age)**2 / (2 * sigma_sum)) / np.sqrt(sigma_sum))

def calibrate_multi(dates, curve):
	# calibrate multiple 14C dates
	# dates = [[lab_code, c14age, uncert], ...]
	#
	# returns distributions = [distribution, ...]; distribution = [p, ...]
	curve_conv_age, curve_uncert = curve[:,1], curve[:,2]
	collect = []
	for _, c14age, uncert in dates:
		collect.append(calibrate(c14age, uncert, curve_conv_age, curve_uncert))
	return collect

