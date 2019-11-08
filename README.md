## Clustering of calibrated radiocarbon dates

## Overview

A python implementation of the method of clustering radiocarbon dates in order to determine whether they represent separate events, or phases in time. For an overview of the method see:

[insert citation of the 2020 paper]

## How to run
<pre><code>python process.py [dates file].txt [sequence / contiguous / overlapping]</code></pre>

[sequence / contiguous / overlapping] specifies the type of OxCal phasing model generated. If no model is specified, sequence is used by default.

Input radiocarbon dates have to be supplied in the text file [dates file].txt where each row represents a date in the format:
<pre><code>[Lab Code], [14C Age], [Uncertainty]</code></pre>
(see example data in [dates_2_events.txt](dates_2_events.txt))

The script generates a graph of Silhouette and p-value and an OxCal model file stored as:
<pre><code>output\[dates file].pdf
output\[dates file].oxcal</code></pre>

Example data representing two events are included in [dates_2_events.txt](dates_2_events.txt). To run with sample data use e.g.:
<pre><code>python process.py dates_2_events.txt sequence</code></pre>


## Requirements

Running the script requires [Python 3.6](https://www.python.org/)

## Dependencies

The script requires the following libraries to be installed:
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/)
<pre><code>pip install numpy
pip install scipy
pip install scikit-learn</code></pre>

## Author:
Peter Demján [peter.demjan@gmail.com](peter.demjan@gmail.com)

Institute of Archaeology of the Czech Academy of Sciences, Prague, v.v.i.

## Acknowledgements:

Development of this script was supported by OP RDE, MEYS, under the project "Ultra-trace isotope research in social and environmental studies using accelerator mass spectrometry", Reg. No. CZ.02.1.01/0.0/0.0/16_019/0000728.

Uses atmospheric data [intcal13.14c](intcal13.14c) from:

Reimer PJ, Bard E, Bayliss A, Beck JW, Blackwell PG, Bronk Ramsey C, Buck CE, Cheng H, Edwards RL, Friedrich M, Grootes PM, Guilderson TP, Haflidason H, Hajdas I, Hatté C, Heaton TJ, Hogg AG, Hughen KA, Kaiser KF, Kromer B, Manning SW, Niu M, Reimer RW, Richards DA, Scott EM, Southon JR, Turney CSM, van der Plicht J. IntCal13 and MARINE13 radiocarbon age calibration curves 0-50000 years calBP. Radiocarbon 55(4). DOI: 10.2458/azu_js_rc.55.16947

## License:
This code is licensed under the [GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) - see the [LICENSE](LICENSE) file for details

