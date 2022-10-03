# Introduction
This is the repository for selection bias modeling project data and plots.
# Contribution
Chenxiao Zeng (The Ohio State University)
Heidi Wu (Boise State University)
Andres Salcedo (University of Arizona)
Chris Hirata (The Ohio State University)
# Structure
/source/corrfunc_CylinderD30_ab.py is the main file containing all methods to analyze the correlation relation in the Abacus data set. Corrfunc is the main package for this analysis.

In /data/abacus_phases, the files are the correlation ouputs either from wp or from xi. The columns for wp are [rp, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm] or [rp, wp_cg, wp_cc, wp_cm, wp_gg, wp_mm] (file name with "_ab", stands for abundance matching). The columns for xi are simiar.
