# extract_features.jl
# This script will take the extracted spectrograms and create a training set
# of data that can be used to train the random forest classifier.
# author: Timothy Evans
# date: 31/07/14

push!(LOAD_PATH, pwd())
importall dataprocessing, config
using Images, DataFrames, ProgressMeter

#load paths from config.jl
specpath,labelledpath,trainpath,modelpath = loadpaths()
p = Progress(length(readdir(labelledpath)),1) # initialise progress bar

println("Loading $(length(readdir(labelledpath))) images and extracting feature vectors ... ")

for spec in readdir(labelledpath)

  #load the images and construct the feature vector for this image pair.
  imgmask = Images.load(string(labelledpath,spec)); # labelled masks are png images to avoid loss
  img = Images.load(string(trainpath,spec[1:end-3],"jpg")); # the corresponding img will be jpg for space reasons
  recorderdata = readtable(string(modelpath,"recorderdata.csv"));
  if (colorspace(imgmask) == "BGRA")
#    print(spec)
    (pixels,freq,boxmu,boxvar,label) = imgfeature(img, imgmask,spec,recorderdata)
  else
    c = colorspace(img)
    # if this error is thrown consider adding elseif conditions for other
    # potential colorspaces
    error("Currently extract_features.jl only considers images with colorspace
    BRGA: received image of type $cs.")
  end

  #concatenate feature data together
  if isdefined(:featuredata)
    featuredata = vcat(featuredata,hcat(pixels,freq,boxmu,boxvar))
    labeldata = vcat(labeldata,label)
  else
    global featuredata = hcat(pixels,freq,boxmu,boxvar);
    global labeldata = label;
  end
  next!(p)
end

print("Subsampling data and writing to $(string(modelpath,datafile)) ... \n")

# subsample data as required
featuredata,labeldata = subsampledata(featuredata,labeldata,subsampleratio)
labeldata = convertclasses(labeldata) # make zeros and ones all zeros and twos into ones

#feature normalise the data
(X,mu,sigma) = featurenormalise(featuredata);

#write data to file and save mu and sigma for future use in prediction
writecsv(string(modelpath,datafile),hcat(X,labeldata))
writecsv(string(modelpath,"musigma.csv"),vcat(mu,sigma))
