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
#p_value = 0.1

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
		uniform = False
		if len(args) > 1:
			if args[1].strip().lower() in ["sequence", "contiguous", "overlapping"]:
				model = args[1].strip().lower()
			elif args[1].strip().lower() == "uniform":
				uniform = True
		if (len(args) > 2) and (args[2].strip().lower() == "uniform"):
			uniform = True
		
		if not os.path.isdir("output"):
			os.makedirs("output")
		
		curve = load_calibration_curve(fcurve, interpolate = False)
		dates = load_dates(fdates)  # [[lab_code, c14age, uncert], ...]
		
		print("\nProcessing %d dates from %s" % (len(dates), fdates))
		
		
		# Randomization testing of the null hypothesis that the observed 14C dates represent a normal / uniform distribution
		
		cal_ages, dists, summed, dists_rnd, sums_rnd, p = get_randomized(dates, curve, uniform = uniform)
		perc_lower = (p_value * 100) / 2
		perc_upper = 100 - perc_lower
		
		sums_rnd_lower, sums_rnd_upper = calc_percentiles(sums_rnd, perc_lower, perc_upper)
		
		if p < p_value:
			null_hypothesis_txt = "Dates are not %s distributed." % ("uniformly" if uniform else "normally")
		else:
			null_hypothesis_txt = "Dates are %s distributed." % ("uniformly" if uniform else "normally")
		
		print()
		print()
		print("p-value:", p)
		print(null_hypothesis_txt)
		print()
		
		fsummed = fdates.split(".")[:-1]
		fsummed[-1] += "_summed"
		fsummed = os.path.join("output", ".".join(fsummed + ["pdf"]))
		
		fig = pyplot.figure(figsize = (15, 4))
		pyplot.fill_between(cal_ages - 1950, sums_rnd_lower, sums_rnd_upper, color = "lightgrey", label = "%0.2f%% of randomized results" % (perc_upper - perc_lower))
		pyplot.plot(cal_ages - 1950, summed, color = "k", label = "Observed dates")
#		idx1, idx2 = calc_range(cal_ages, summed / summed.sum(), 0.9999)
		idxs = np.where(sums_rnd_upper > 0)[0]
		idx1, idx2 = idxs.min(), idxs.max()
		pyplot.xlim(cal_ages[int(idx1)] - 1950, cal_ages[int(idx2)] - 1950)
#		pyplot.xlim(3600 - 1950, 2200 - 1950)  # DEBUG 6 events
#		pyplot.xlim(3300 - 1950, 2700 - 1950)  # DEBUG 2 events
#		pyplot.xlim(7300 - 1950, 6750 - 1950)  # DEBUG herxheim
		pyplot.gca().invert_xaxis()
		pyplot.xlabel("Calendar age (yrs BC)")
		pyplot.ylabel("Summed p")
		pyplot.annotate("p: %0.5f\n%s" % (p, null_hypothesis_txt), xy = (0.05, 0.95), xycoords = "axes fraction", fontsize = 12, horizontalalignment = "left", verticalalignment = "top")
		pyplot.legend()
		pyplot.tight_layout()
		pyplot.savefig(fsummed)
		fig.clf()
		pyplot.close()
		
		# Calculate clustering
		
		clusters, means, ps, sils = get_clusters_hca(dates, curve, uniform = uniform)
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
		
		txt = "Clusters_n,Phase,Mean (Cal. yrs BP)\n"
		for n in means:
			labels = sorted(list(means[n].keys()), key = lambda label: means[n][label])[::-1]
			for idx, label in enumerate(labels):
				txt += "%d,%d,%0.2f\n" % (n, idx + 1, means[n][label])
		with open(fmeans, "w") as f:
			f.write(txt)

