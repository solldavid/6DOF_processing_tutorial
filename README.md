# Seismological processing of six degree-of-freedom ground-motion data

In this tutorial, you will learn how to process six degree-of-freedom ground-motion data comprised of three components of translational motion and three components of rotational motion. The data used in this tutorial was recorded on the large ring laser gyroscope ROMY in Germany (http://www.romy-erc.eu), after the 2018 M7.9 gulf of Alaska earthquake as described in the accompanying paper:  

Sollberger, D., Igel, H., Schmelzbach, C., Edme, P., van Manen, D.-J., Bernauer, F., Yuan, S., Wassermann, J., Schreiber, U., and Robertsson, J. O. A. (2020): **Seismological processing of six degree-of-freedom ground-motion data**, *Sensors*.

Please refer to this paper for further details on the algorithm used for the analysis. Additional information on 6DOF polarization analysis can be found in the following publication:

Sollberger, D., Greenhalgh, S. A., Schmelzbach, C., Van Renterghem, C., and Robertsson, J. O. A. (2018): **6-C polarization analysis using point measurements of translational and rotational ground-motion: theory and applications**, *Geophysical Journal International*, 213(1), https://doi.org/10.1093/gji/ggx542.

Note that at the time of publication, the underlying code used in this analysis is still under development and some features might not yet work as intended. If you encounter any bugs or problems, please report them. You can also directly propose changes via a pull request. Any queries about this code should be directed to David Sollberger, Institute of Geophysics, ETH Zurich (david.sollberger@erdw.ethz.ch). 

## Prerequisites
The code requires the following prerequisites:
- Obspy (Python framework for processing of seismological data)
- PyTables
- tqdm (Progress bar)
- Jupyter (to run the notebook)

We recommend to install them using anaconda in a new environment via:

$ conda config --add channels conda-forge # This adds the conda-forge channel to your Anaconda configuration

$ conda create -n 6DOF python=3.7 

$ conda activate 6DOF

$ conda install obspy pytables tqdm jupyter
