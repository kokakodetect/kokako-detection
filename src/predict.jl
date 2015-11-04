# predict.jl
# This script takes spectrograms and computes the corresponding predictions
# author: Timothy Evans
# date: 04/08/14

push!(LOAD_PATH, pwd())
importall dataprocessing, config
using Images, DataFrames, DecisionTree, JLD, BoundingBoxes, ProgressMeter

#load paths from config.jl
(specpath,labelledpath,trainpath,modelpath) = loadpaths();

#add in extract spectrograms here

#load in Random Forest classifier
println("loading in Random Forest classifier...")
model = jldopen(string(modelpath,"model.jld"),"r") do file
  read(file,"model")
end

#load saved mean and standard dev for normalisation
normvalues = readcsv(string(modelpath,"musigma.csv"));
mu = normvalues[1,:];
sigma = normvalues[2,:];
nsyllables = 0;
p = Progress(length(readdir(specpath)),1) # initialise progress bar

# check if the syllables.csv file is empty and prompt user
if filesize(string(modelpath,"syllables.csv")) > 0
  doappend = syllablesfileprompt();
  if ~doappend
    #delete the file
    rm(string(modelpath,"syllables.csv"))
  end
end


for spec in readdir(specpath)

  #load the image and corresponding recorder data
  img = Images.load(string(trainpath,spec));
  recorderdata = readtable(string(modelpath,"recorderdata.csv"));

  #extract features from the image
  (pixels,freq,boxmu,boxvar) = imgfeature_predict(img,spec,recorderdata);
  featuredata = hcat(pixels,freq,boxmu,boxvar);

  # normalise the features according to training mu and sigma
  (m,n) = size(featuredata);
  X = (featuredata - ones(m)*mu) ./ (ones(m)*sigma);

  #apply the random forest classifier and reshape the predictions into a matrix
  y = apply_forest(model,X);
  predimg = convert(Array{Float64,2},reshapepred(y,height(img),width(img)));
  predimg = convthresh(predimg);

  #extract all syllables with area greater than minarea as defined in config.jl
  (mask,area,boxes) = segmentimg(predimg);
  nsyllables += length(area)

  # if the spectrogram had segments then save the bounding boxes to file
  if ~isempty(boxes)
    (nrows,ncols) = size(mask)
    dumpsyllables(boxes,modelpath,spec,recorderdata,nrows,ncols)
  end

  #update progress bar after each image has been segmented
  next!(p)

end

println("Found $nsyllables syllables")
