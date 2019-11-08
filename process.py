import sys
import os
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot

from fnc_common import *
from fnc_cluster import *
from fnc_oxcal import *

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
		
		# Calculate clustering
		
		clusters, means, ps, sils = get_clusters_hca(dates, curve, pool)
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
		ax2.plot([clu_ns[0], clu_ns[-1]], [0.05, 0.05], "--", color = color2, linewidth = 0.7)
		ax2.annotate("p = 0.05", xy = (clu_ns.mean(), 0.05), xytext = (0, -3), textcoords = "offset pixels", va = "top", ha = "center", color = color2)
		ax2.tick_params(axis = "y", labelcolor = color2)
		
		pyplot.xticks(clu_ns, clu_ns)
		
		fig.tight_layout()
		pyplot.savefig(fgraph)
		
		# Find optimal number of clusters based on Silhouette and specified p-value and generate OxCal phasing model
		
		foxcal = os.path.join("output", ".".join(fdates.split(".")[:-1] + ["oxcal"]))
		
		n_opt = get_opt_clusters(clusters, ps, sils, p_value)
		
		print("\n\nFound optimal no. of clusters: %d\n" % (n_opt))
		print("Generated Silhouette graph: %s" % (fgraph))
		print("Generated OxCal model file: %s" % (foxcal))
		
		txt = gen_oxcal(clusters[n_opt], means[n_opt], dates, curve, model)
		
		with open(foxcal, "w") as f:
			f.write(txt)

