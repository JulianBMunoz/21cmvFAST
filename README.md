# 21cmvFAST (04/16/19)
A modified version of 21cmFAST (through 21CMMC) that includes the effect of the 
dark matter (DM)-baryon relative velocities. 
By Julian B Munoz, 
based on the original 21cmFAST + 21CMMC written by Bradley Greig and Andrei Mesinger.
Check out their Gits at:
https://github.com/andreimesinger/21cmFAST
and
https://github.com/BradGreig/21CMMC.

The main purpose of this goal is to show that including DM-b relative velocities 
produces velocity-induced acoustic oscillations (VAOs) in the 21-cm power spectrum.
Additionally, it allows 21cmFAST to properly account for molecular-cooling haloes, 
which are expected to drive star formation during cosmic dawn,
as both relative velocities and Lyman-Werner feedback, suppress halo formation, 
which here we include.
This yields accurate 21-cm predictions all the way to reionization (z>~10).


The usage is the same as standard 21CMMC, that is:

1- go to Programs to install (change the Makefile if needed) and
make
2- create boxes of initial conditions:
./init 1. 1.
which generates boxes (both delta and vcb) with the cosmology of WalkerCosmo_1... 
3- evolve the boxes to the desired redshift Z (which saves the intermediate ones):
./create_dens_boxes_for_LC 1. 1. Z
4- Finally, run the 21cmFAST driver:
./drive_21cmMC_streamlined 1. 1. 0 0 0 Z
and that will do it.

Feel free to go into
Parameter_files/COSMOLOGY.H 
to vary the "OUTPUT_FOLDER" name, and the velocity/feedback parameters, and into
Parameter_files/INIT_PARAMS.H 
to change the size and resolution of the boxes (remember to do ./init, etc. after changing any init settings).
You can change "PRINT_COEVAL_21cmBoxes" within Parameter_files/Variables.H to save 21cm maps (in addition to power spectra).


In a nutshell, these are the main changes to 21cmFAST:
-We update the base code by reading transfer functions from CLASS 
(included as External_tables/Transfers_z0.dat for LCDM cosmology),
both for better precision than the analytic approximations used before, and to include velocities.
-We generate a box of initial DM-b velocities, keeping the correlation with over/underdensities.
-We use an interpolator for Fcoll and sigma_cool as a function of redshift, \delta, and vcb,
which is calculated elsewhere for computational speed, and all that 21cmvFAST requires are the tables:
External_tables/Fcollapse_table_F.dat
External_tables/sigmacool_table_F.dat
where F is an integer that describes the feedback strength. 
F=0 is no feedback, 1 is a very-low feedback model (unused), and 2-4 are (low,regular,high) as described in the paper.
F=10 is atomic-cooling only.
-We do not alter the ionization part, only the X-rays and Lyman-coupling.


There are more details about the implementation in
Documentation/Modifications_to_21cmvFAST.txt


If you use this work please cite:

- 21cmvFAST:
https://arxiv.org/abs/1904.XXXXX
https://arxiv.org/abs/1904.XXXXX

- Original 21cmFAST
https://arxiv.org/abs/1003.3878

- Original 21CMMC:
https://arxiv.org/abs/1501.06576
https://arxiv.org/abs/1705.03471
https://arxiv.org/abs/1801.01592

and please refer to the Gits of 21cmFAST,21cmmc (linked above), and 21cmvFAST:
https://github.com/JulianBMunoz/21cmvFAST
