import sys
import os
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot

from fnc_common import *
from fnc_cluster import *
from fnc_oxcal import *
from fnc_simulate import *

fcurve = "intcal13.14c"

p_value = 0.05

if __name__ == '__main__':
	
	args = []
	if len(sys.argv) > 1:
		args = [sys.argv[idx] for idx in range(1, len(sys.argv))]
	if not args:
		print('''
Dates file not specified.
Command line syntax: \"python process.py [dates file].txt [sequence / contiguous / overlapping]\"
\tsequence / contiguous / overlapping specifies the type of OxCal phasing model generated.
\tIf no model is specified, sequence is used by default.
		''')
	else:
		fdates = args[0].strip()
		model = "sequence"
		if (len(args) > 1) and (args[1].strip().lower() in ["sequence", "contiguous", "overlapping"]):
			model = args[1].strip().lower()
		
		if not os.path.isdir("output"):
			os.makedirs("output")
		
		curve = load_calibration_curve(fcurve)
		dates = load_dates(fdates)  # [[lab_code, c14age, uncert], ...]
		
		print("\nProcessing %d dates from %s" % (len(dates), fdates))
		
		pool = mp.Pool(processes = mp.cpu_count())
		
		
		# Compare observed and randomized summed dates
		
		cal_ages, dists, summed, dists_rnd, sums_rnd = get_randomized(dates, curve, pool, p_diff_max = 0.001)
		perc_lower = (p_value * 100) / 2
		perc_upper = 100 - perc_lower
		
		sums_rnd_lower, sums_rnd_upper = calc_percentiles(sums_rnd, perc_lower, perc_upper)
		pdiff = 0
		mask = (summed > sums_rnd_upper)
		if mask.any():
			pdiff += (summed[mask] - sums_rnd_upper[mask]).sum()
		mask = (summed < sums_rnd_lower)
		if mask.any():
			pdiff += (sums_rnd_lower[mask] - summed[mask]).sum()
		if pdiff > 0:
			single_event_txt = "Dates represent multiple events."
		else:
			single_event_txt = "Dates represent a single event."
		
		fsummed = fdates.split(".")[:-1]
		fsummed[-1] += "_summed"
		fsummed = os.path.join("output", ".".join(fsummed + ["pdf"]))
		
		fig = pyplot.figure(figsize = (15, 4))
		pyplot.fill_between(cal_ages - 1950, sums_rnd_lower, sums_rnd_upper, color = "lightgrey", label = "%0.2f%% of randomized results" % (perc_upper - perc_lower))
		pyplot.plot(cal_ages - 1950, summed, color = "k", label = "Observed dates")
		pyplot.gca().invert_xaxis()
		pyplot.xlabel("Calendar age (yrs BC)")
		pyplot.ylabel("Summed p")
		pyplot.annotate("Summed p outside interval of randomness = %0.3f\n%s" % (pdiff, single_event_txt), xy = (0.05, 0.95), xycoords = "axes fraction", horizontalalignment = "left", verticalalignment = "top")
		pyplot.legend()
		pyplot.tight_layout()
		pyplot.savefig(fsummed)
		fig.clf()
		pyplot.close()
		
		print()
		print("Summed p outside interval of randomness = %0.3f\n%s" % (pdiff, single_event_txt))
		print()
		
		
		# Calculate clustering
		
		clusters, means, ps, sils = get_clusters_hca(dates, curve, pool, p_diff_max = 0.001)
		# clusters = {n: {label: [idx, ...], ...}, ...}; n = number of clusters, idx = index in dates
		# means = {n: {label: mean, ...}, ...}; mean = mean of the summed distributions of the calibrated dates within the cluster
		# ps = {n: p-value, ...}; p-value of the null hypothesis that the Silhouette for n clusters is the product of randomly distributed dates
		# sils = {n: Silhouette, ...}
		
		# Plot Silhouette and p-value for solutions with different numbers of clusters
		
		clu_ns = np.array(sorted(list(clusters.keys())), dtype = int)
		ps_plot = np.array([ps[clu_n] for clu_n in clu_ns])
		sils_plot = np.array([sils[clu_n] for clu_n in clu_ns])
		
		fgraph = os.path.join("output", ".".join(fdates.split(".")[:-1] + ["pdf"]))
		
		color1 = "blue"
		color2 = "green"
		
		fig, ax1 = pyplot.subplots()
		
		ax1.set_xlabel("Clusters")
		ax1.set_ylabel("Mean Silhouette Coefficient", color = color1)
		ax1.plot(clu_ns, sils_plot, color = color1)
		ax1.plot(clu_ns, sils_plot, ".", color = color1)
		ax1.tick_params(axis = "y", labelcolor = color1)
		
		ax2 = ax1.twinx()
		ax2.set_ylabel("p", color = color2)
		ax2.plot(clu_ns, ps_plot, color = color2)
		ax2.plot(clu_ns, ps_plot, ".", color = color2)
		ax2.plot([clu_ns[0], clu_ns[-1]], [p_value, p_value], "--", color = color2, linewidth = 0.7)
		ax2.annotate("p = %0.3f" % (p_value), xy = (clu_ns.mean(), p_value), xytext = (0, -3), textcoords = "offset pixels", va = "top", ha = "center", color = color2)
		ax2.tick_params(axis = "y", labelcolor = color2)
		
		pyplot.xticks(clu_ns, clu_ns)
		
		fig.tight_layout()
		pyplot.savefig(fgraph)
		fig.clf()
		pyplot.close()
		
		# Find optimal number of clusters based on Silhouette and specified p-value and generate OxCal phasing model
		
		foxcal = os.path.join("output", ".".join(fdates.split(".")[:-1] + ["oxcal"]))
		
		n_opt = get_opt_clusters(clusters, ps, sils, p_value)
		
		if n_opt is None:
			n_opt = 0
		
		print("\n\nFound optimal no. of clusters: %d\n" % (n_opt))
		print("Generated Silhouette graph: %s" % (fgraph))
		
		if n_opt > 0:
			print("Generated OxCal model file: %s" % (foxcal))
			txt = gen_oxcal(clusters[n_opt], means[n_opt], dates, curve, model)
			
			with open(foxcal, "w") as f:
				f.write(txt)
		
		fmeans = fdates.split(".")[:-1]
		fmeans[-1] += "_means"
		fmeans = os.path.join("output", ".".join(fmeans + ["csv"]))
		print("Generated Means file: %s" % (fmeans))
		
		txt = "Clusters_n,Label,Mean (Cal. yrs BP)\n"
		for n in means:
			for label in means[n]:
				txt += "%d,%d,%0.2f\n" % (n, label, means[n][label])
		with open(fmeans, "w") as f:
			f.write(txt)

