This repository contains codes for the Geometric Scattering Trajectory Homology (GSTH) method from the paper "Tissue-wide coordination of calcium signaling regulates the epithelial stem cell pool during homeostasis".


###Intoduction
GSTH is a method used to learn and visualize patterns from calcium signaling data, which combines data geometry and topology. GSTH models Ca^{2+} as signals over a cell adjacency graph and uses a multi-level wavelet-like transform (called a scattering transform) to extract signaling patterns from our high dimensional \textit{in vivo} datasets.
To run the codes. the following packages need to be installed:
1. Numpy
2. Scipy
3. Networkx
4. Eiren
5. PHATE

#### Installation with `pip`

The Python version of PHATE can be installed by running the following from a terminal:

    pip install --user phate

Installation of PHATE and all dependencies should take no more than five minutes.

