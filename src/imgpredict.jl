# imgpredict.jl
# This script takes a single image and runs the random forest predictor on it
# author: Timothy Evans
# date: 04/08/14

push!(LOAD_PATH, pwd())
using DecisionTree, JLD, Images, DataFrames, Winston.imagesc, Winston.figure
importall config, dataprocessing

#load paths from config.jl
specpath,labelledpath,trainpath,modelpath = loadpaths()

println("Loading Random Forest...")
model = jldopen(string(modelpath,"model.jld"),"r") do file
  read(file,"model")
end

spec = "BSR01_140107_0_123.jpg"#readdir(specpath)[1];
print("Loading image $spec")
img = Images.load(string(trainpath,spec));
imgmask = Images.load(string(labelledpath,spec[1:end-3],"png"));
recorderdata = readtable(string(modelpath,"recorderdata.csv"));
(pixels,freq,boxmu,boxvar,label) = imgfeature(img, imgmask,spec,recorderdata);
featuredata = hcat(pixels,freq,boxmu,boxvar);

#normalise features using saved mean and standard dev
normvalues = readcsv(string(modelpath,"musigma.csv"));
mu = normvalues[1,:];
sigma = normvalues[2,:];
(m,n) = size(featuredata);
X = (featuredata - ones(m)*mu) ./ (ones(m)*sigma);

y = apply_forest(model,X);
ytest = convertclasses(label); # make zeros and ones all zeros and twos into ones

#
gt = reshapepred(ytest,width(img),height(img));
predimg = convert(Array{Float64,2},reshapepred(y,width(img),height(img)));
predimg = convthresh(predimg)
figure()
imagesc(gt)
figure()
imagesc(predimg)

confm = confusmat(ytest,y);
acc = trace(confm)./length(ytest);
C = confm./sum(confm,2);
@printf "\nConfusion Matrix:\n\n\t[%.3f , %.3f\n\t%.3f , %.3f]\n\nConfusion Accuracies:\n\n\t[%d , %d\n\n\t %d , %d]\n\nTest Accuracy: %.3f" C[1,1] C[1,2] C[2,1] C[2,2] confm[1,1] confm[1,2] confm[2,1] confm[2,2] acc
