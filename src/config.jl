module config

# This is the main configuration file for the whole of the kokako-detect project.
# Any constants and parameters that need to be changed must be changed here only.
# Likewise the addition of any such parameters should also be added here.
# Author: Tim Evans
# Date: 30/07/14


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# paths and constants pertaining to extracting features
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

export wavdir, traindir, speclength, path_rel, test_size, specwidth, labelleddir,datafile,
ntrees, ratio, subsampleratio, modeldir, specdir,boxwidth,boxheight,thresh,kernwidth,kernsigma,
minarea

#the following paths are all relative to the project directory and will be concatenated
#To use absolute paths sets path_rel = false
path_rel = true
wavdir = "data/wavs/"
traindir = "data/training/"
specdir = "data/spectrograms/"
labelleddir = "data/labelled/"
datafile = "data.csv"

#------------------------------------------------------------------------------
#                           filter parameters
#------------------------------------------------------------------------------
fc = 1250
npoles = 4
speclength = 10 #seconds
specwidth = 930 #setting the width of spectrograms produced

#------------------------------------------------------------------------------
#                       feature extraction paramaters
#------------------------------------------------------------------------------
boxwidth = 3
boxheight = 10
# This determines how much we will subsample the noise and labelled noise in our
# dataset. E.g. if we have 30 observations of bird call pixels, and subsampleratio = 0.5
# then we will add 15 labelled noise, and 15 unlabelled noise to the dataset
subsampleratio = 1

#------------------------------------------------------------------------------
#                          Random Forest parameters
#------------------------------------------------------------------------------
modeldir = "data/model/"
ntrees = 10;
ratio = 0.2;

#------------------------------------------------------------------------------
#                prediction parameters (post-processing)
#------------------------------------------------------------------------------
thresh = 0.6
kernwidth = 5 # Gaussian kernel parameters
kernsigma = 2
minarea = 16 # minimum sized syllable to consider as a bird

end
