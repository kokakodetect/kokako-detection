# train.jl
# This script takes the saved feature data and will train a Random Forest to
# classify bird syllables in spectrograms
# author: Timothy Evans
# date: 02/08/14

push!(LOAD_PATH, pwd())
using DecisionTree, JLD
importall config, dataprocessing

#load paths from config.jl
specpath,labelledpath,trainpath,modelpath = loadpaths()
print("Loading in training data...\n")
#import data
data = readcsv(string(modelpath,datafile));
(m,n) = size(data);

#use 70% of data in training set
ind = randperm(m);
ntrain = Int(ceil(0.7*m));
trainind = ind[1:ntrain];
testind = ind[ntrain+1:end];

xtrain = data[trainind,1:end-1];
ytrain = toint(data[trainind,end]);
xtest = data[testind,1:end-1];
ytest = toint(data[testind,end]);

print("Training Random Forest...\n")
# build the random forest using config paramaters.
model = build_forest(ytrain, xtrain, Int(ceil(sqrt(n))), ntrees, ratio)

print("Testing model accuracy...\n")
#pixelwise accuracy:
testpred = toint(apply_forest(model,xtest));
# accuracy = 1-sum(abs(ytest-testpred))/length(ytest);

confm = confusmat(ytest,testpred);
acc = trace(confm)./length(ytest);
C = confm./sum(confm,2);

@printf "\nConfusion Matrix:\n\n\t[%.3f , %.3f\n\t%.3f , %.3f]\n\nConfusion Accuracies:\n\n\t[%d , %d\n\t %d , %d]\n\nTest Accuracy: %.3f\n" C[1,1] C[1,2] C[2,1] C[2,2] confm[1,1] confm[1,2] confm[2,1] confm[2,2] acc

print("Writing Random Forest model to disk (if ntrees is big this may take some time)...\n")
# Save the random forest structure for future use
jldopen(string(modelpath,"model.jld"),"w") do file
  write(file,"model",model)
end
