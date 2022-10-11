import numpy as np

from fnc_common import *
from fnc_sum import *

def gen_dates(dates):
	
	txt = ""
	for lab_code, c14age, uncert in dates:
		txt += '''
				R_Date("%s", %d, %d);
		''' % (lab_code, c14age, uncert)
	return txt

def gen_sequence(data):
	
	txt = ""
	for phase in sorted(list(data.keys())):
		txt += '''
		Boundary("Start %d");
		Phase("%d")
		{
			%s
		};
		Boundary("End %d");
		''' % (phase, phase, gen_dates(data[phase]), phase)
	
	return '''
	Sequence()
	{
		%s
	};
	''' % (txt)
	
def gen_contiguous(data):
	
	txt = ""
	last_phase = None
	for phase in sorted(list(data.keys())):
		if last_phase is None:
			txt += '''
		Boundary("Start %d");
			''' % (phase)
		else:
			txt += '''
		Boundary("Transition %d/%d");
			''' % (last_phase, phase)
		txt += '''
		Phase("%d")
		{
			%s
		};
		''' % (phase, gen_dates(data[phase]))
		last_phase = phase
	txt += '''
		Boundary("End %d");
	''' % (last_phase)
	
	return '''
	Sequence()
	{
		%s
	};
	''' % (txt)

def gen_overlapping(data):
	
	txt = ""
	for phase in sorted(list(data.keys())):
		txt += '''
		Sequence()
		{
			Boundary("Start %d");
			Phase("%d")
			{
				%s
			};
			Boundary("End %d");
		};
		''' % (phase, phase, gen_dates(data[phase]), phase)
	return '''
	Phase()
	{
		%s
	};
	''' % (txt)

def gen_none(data):
	
	txt = ""
	for phase in sorted(list(data.keys())):
		txt += '''
		Label("Cluster %d");
		%s
		''' % (phase, gen_dates(data[phase]))
	return txt

def gen_oxcal(clusters, means, dates, curve, model):
	# Generate OxCal phasing model based on clustering
	# clusters = {label: [idx, ...], ...}; idx = index in dates
	# means = {label: mean, ...}
	# model = "sequence" / "contiguous" / "overlapping" / "none"
	
	distributions = calibrate_multi(dates, curve)
	date_means = np.array([calc_mean_std(curve[:,0], dist)[0] for dist in distributions])
	
	labels = sorted(list(clusters.keys()), key = lambda label: means[label])[::-1]
	
	data = {} # {phase: [[lab_code, c14age, uncert], ...], ...}
	phase = 1
	for label in labels:
#		data[phase] = [dates[idx] for idx in sorted(clusters[label], key = lambda idx: date_means[idx])[::-1]]
		data[phase] = [dates[idx] for idx in sorted(clusters[label], key = lambda idx: dates[idx][1])[::-1]]  # DEBUG
		phase += 1
	txt = {
		"sequence": gen_sequence,
		"contiguous": gen_contiguous,
		"overlapping": gen_overlapping,
		"none": gen_none,
	}[model](data)
	return '''
Plot()
{
	%s
};
	''' % (txt)
