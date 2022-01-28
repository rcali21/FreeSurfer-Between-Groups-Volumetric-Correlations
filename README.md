# **FreeSurfer-Between-Groups-Volumetric-Correlation**
* Statistical pipeline for correlating volumetric data from FS segmentation between two groups using recon-all stats file.
* Generates a line of best-fit for specific anatomical locations of interest. (based on FreeSurfer aseg parcellation)



## Using the pipeline
* Aseg data needs to be extracted from each of the groups you're looking to examine volumetric data in. 
* There are multiple ways to pull this data, the easiest being through the use of asegstats2table tool in FreeSurfer.
