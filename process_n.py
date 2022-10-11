import sys
import os
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot

from fnc_common import *
from fnc_cluster import *
from fnc_oxcal import *
from fnc_simulate import *

fcurve = "intcal20.14c"

if __name__ == '__main__':
	
	args = []
	if len(sys.argv) > 2:
		args = [sys.argv[idx] for idx in range(1, len(sys.argv))]
	if not args:
		print('''
Dates file not specified.
Command line syntax: \"python process_n.py [dates file].txt [clusters_n] [sequence / contiguous / overlapping / none]\"
\tclusters_n = number of clusters to extract
\tsequence / contiguous / overlapping specifies the type of OxCal phasing model generated.
\tIf no model is specified, sequence is used by default.
		''')
	else:
		fdates = args[0].strip()
		clusters_n = int(args[1].strip())
		model = "sequence"
		if (len(args) > 2) and (args[2].strip().lower() in ["sequence", "contiguous", "overlapping", "none"]):
			model = args[2].strip().lower()
		
		if not os.path.isdir("output"):
			os.makedirs("output")
		
		curve = load_calibration_curve(fcurve, interpolate = False)
		dates = load_dates(fdates)  # [[lab_code, c14age, uncert], ...]
		
		print("\nProcessing %d dates from %s" % (len(dates), fdates))
		
		# Calculate clustering
		
		clusters, means = get_n_clusters_hca(dates, curve, clusters_n)
		# clusters = {label: [idx, ...], ...}; idx = index in dates
		# means = {label: mean, ...}; mean = mean of the summed distributions of the calibrated dates within the cluster
		
		
		# Generate OxCal phasing model
		
		fname = fdates.split(".")[:-1]
		fname[-1] = "%s_n_%d" % (fname[-1], clusters_n)
		
		foxcal = os.path.join("output", ".".join(fname + ["oxcal"]))
		print("Generated OxCal model file: %s" % (foxcal))
		txt = gen_oxcal(clusters, means, dates, curve, model)
		
		with open(foxcal, "w") as f:
			f.write(txt)
		
		fmeans = fname.copy()
		fmeans[-1] += "_means"
		fmeans = os.path.join("output", ".".join(fmeans + ["csv"]))
		print("Generated Means file: %s" % (fmeans))
		
		txt = "Phase,Mean (Cal. yrs BP)\n"
		labels = sorted(list(means.keys()), key = lambda label: means[label])[::-1]
		for idx, label in enumerate(labels):
			txt += "%d,%0.2f\n" % (idx + 1, means[label])
		with open(fmeans, "w") as f:
			f.write(txt)

