module dataprocessing

# This module contains all of the necessary functions for the kokako-detect
# project. Any additional functions should be added here.
# author: Timothy Evans
# date: 30/07/14


export filterwav, dumpspectrograms, imgfeature, featurenormalise, subsampledata, convertclasses,
toint, confusmat, imgfeature_predict, reshapepred, convthresh, loadpaths, segmentimg, Bounds2,
dumpsyllables, syllablesfileprompt
using DSP, Images, DataFrames, BoundingBoxes,config

#define our use of bounding boxes in x and y coordinates
@boundingbox Bounds2, "x", "y"

function filterwav(signal, fs)
  #filter the wav recording using a high pass Butterworth filter
  #with cutoff frequency 1250Hz
  responsetype = Highpass(fc/(fs/2))
  designmethod = Butterworth(npoles) # fourth order filter
  signal = filt(digitalfilter(responsetype,designmethod),signal)
  return  signal
end

function dumpspectrograms(signal,segments,path,nameprefix,specwidth)
  left = segments[1:end-1]
  right = segments[2:end]-1
  for (i,l) in enumerate(left)
    #compute a spectrogram with roughly equal time and freq information
    spec = spectrogram(signal[l:right[i]],specwidth,window=hanning)
    filename = string(path,nameprefix,"$i.png")
    imwrite(flipud(spec.power/maximum(spec.power)),filename)
  end
end

function imgfeature(img,imgmask,filename,recorderdata)
#construct feature histogram for each pixel in image and corresponding label

  sepmask = separate(imgmask).data; #separate layers in order to get layers

  #get the dimensions of the img and determine how many pixels will be
  #lost by the boxwidth and boxheight limitation
  nrows = height(img);
  ncols = width(img);
  npix = (2*boxheight+1)*(2*boxwidth+1); #number of pixels in a boxheight
  nbox = (ncols - 2*boxwidth)*(nrows - 2*boxheight); #number of boxes in an image

  #initialise variables
  boxmu = Array(Float64,nbox);
  boxvar = Array(Float64,nbox);
  freq = Array(Float64,nbox);
  pixels = Array(Float64,nbox,npix); #initialise the data array to correct dims
  label = zeros(nbox);
  pcount = 0;

  # get the frequency vector for the spectrogram
  freqvector = getfreqvector(getrecorder(filename),recorderdata,nrows);

  #extract values from Gray image structure
  imgdata = getgrayval(img);
  for row in (boxheight+1):(nrows-boxheight)
    for col in (boxwidth+1):(ncols-boxwidth)
      pcount += 1

      # determine the feature histogram for each pixel
      box = imgdata[(col-boxwidth):(col+boxwidth),(row-boxheight):(row+boxheight)]
      boxmu[pcount] = mean(box);
      boxvar[pcount] = var(box);
      freq[pcount] = freqvector[row];
      pixels[pcount,:] = box[:]';

      #get the corresponding label
      label[pcount] = getlabel(sepmask,row,col)
    end
  end
  return pixels,freq,boxmu,boxvar,label
end

function imgfeature_predict(img,filename,recorderdata)
#construct feature histogram for each pixel in image and corresponding label

  #get the dimensions of the img and determine how many pixels will be
  #lost by the boxwidth and boxheight limitation
  nrows = width(img);
  ncols = height(img);
  npix = (2*boxheight+1)*(2*boxwidth+1); #number of pixels in a boxheight
  nbox = (ncols - 2*boxwidth)*(nrows - 2*boxheight); #number of boxes in an image

  #initialise variables
  boxmu = Array(Float64,nbox);
  boxvar = Array(Float64,nbox);
  freq = Array(Float64,nbox);
  pixels = Array(Float64,nbox,npix); #initialise the data array to correct dims
  pcount = 0;

  #get the frequency vector for this spectrogram
  freqvector = getfreqvector(getrecorder(filename),recorderdata,nrows);

  #extract values from Gray image structure
  imgdata = getgrayval(img);
  for row in (boxheight+1):(nrows-boxheight)
    for col in (boxwidth+1):(ncols-boxwidth)
      pcount += 1

      # determine the feature histogram for each pixel
      box = imgdata[ (row-boxheight):(row+boxheight) , (col-boxwidth):(col+boxwidth)]
      boxmu[pcount] = mean(box);
      boxvar[pcount] = var(box);
      freq[pcount] = freqvector[row];
      pixels[pcount,:] = box[:]';
    end
  end
  return pixels,freq,boxmu,boxvar
end

function getlabel(imgmask,r,c)

  if imgmask[r,c,1] == 1
    label = 1 # pixel labelled as a non-target
  elseif imgmask[r,c,3] == 1
    label = 2 # pixel labelled as a target syllable
  else
    label = 0 # pixel unlabelled
  end

  return label

end

function getgrayval(img)
  #extract the floating point value for the grayscale image.
  m = width(img)
  n = height(img)
  A = Array(Float64,m,n)  #array of values
  for i in 1:m
    for j in 1:n
      A[i,j] = convert(Float64,img[i,j].val); #convert to floats
    end
  end
  return A
end

function featurenormalise(data)
#normailse given data by (data - mu)/sigma and return mu and sigma

  mu = mean(data,1) # find mean and std of columns
  sigma = std(data,1)
  (m,n) = size(data)
  X = (data - ones(m)*mu) ./ (ones(m)*sigma)
  return X,mu,sigma
end

function featurenormalise_predict(data,mu,sigma)
#normailse given data by (data - mu)/sigma and return mu and sigma

  X = (data - ones(m)*mu) ./ (ones(m)*sigma)
  return X
end

function getrecorder(filename)
# parse the filename and extract the filename and return the recorder
  recorder = filename[1:search(filename,'_')-1];
  return recorder
end

function getfreqvector(recorder,recorderdata,n)
#given the recorder name return the corresponding frequency axis
#of the spectrograms for this recorder
  fs=0;
  count = 0;

  #search for the recorder in the data
  for name in recorderdata[:recorders]
    count += 1;
    if (name == recorder)
      #found the recorder, extract the sampling frequency
      fs = recorderdata[:fsample][count];
    end
  end
  if fs == 0
    #throw error if the recorder is not found
    error("Sampling frequency must be nonzero. Recorder $recorder not found in data.")
  end

  #order the frequency vector backwards due to image indexing
  freq = linspace(fs/2,0,n+1)[1:end-1];
  return freq
end

function subsampledata(featuredata,labeldata,ratio)

  #find the indices of each the three different labels
  nzind = find(labeldata);
  ind0 = setdiff(1:length(labeldata),nzind);
  ind1 = nzind[setdiff(1:length(nzind), find(labeldata[nzind]-1))]
  ind2 = setdiff(nzind,ind1)

  M2 = length(ind2);

  # M is the number of each type we will subsample. we should always have more
  # noise than bird calls but the min is there for robustness
  M0 = Int(min(length(ind0),floor(M2*ratio)))
  M1 = Int(min(length(ind1),floor(M2*ratio)))

  #random subsample indexing vectors
  subind0 = ind0[randperm(length(ind0))[1:M0]]
  subind1 = ind1[randperm(length(ind1))[1:M1]]

  #put all indexing vectors together and extract subsample
  ind = vcat(subind0,subind1,ind2)
  ind = ind[randperm(length(ind))] #randomise the indices up so they arent ordered
  featuredata = featuredata[ind,:]
  labeldata = labeldata[ind]

  return featuredata, labeldata

end

function convertclasses(labels)
# this function converts all negative classes (ie. 0's and 1's) to 0's and
# the positive class (2's) to 1's for the sake of the classifier training.

  labels = labels - 1
  for i in 1:length(labels)
    if labels[i] < 0
      labels[i] = 0
    end
  end
  return labels
end

function toint(x)
# possibly unnecessary in the future/now but language had no convert function
# for arrays. Simple loop through array conversion.

  cx = Array(Int,size(x))
  for i in 1:length(x)
    cx[i] = convert(Int,x[i]);
  end
  return cx
end

function confusmat(gt,pred)
# create a 2 class confusion matrix
  N = length(gt)
  C = zeros(2,2)
  nn,np,pn,pp = 0,0,0,0;
  length(pred) == N || throw(error("Inconsistent lengths."))
  for i in 1:N
    if gt[i] == 0
      if pred[i] == 0
        nn += 1 #add a correct negative
      else
        np += 1 #add a false positive
      end
    else
      if pred[i] == 0
        pn += 1 # add a false negative
      else
        pp += 1 # add a correct positive
      end
    end
  end
  C = hcat([nn,np],[pn,pp]); #putting things together
  return C
end

function reshapepred(y,m,n)
  cols = m - 2*boxwidth;
  rows = n - 2*boxheight;
  return reshape(y,cols,rows)
end

function convthresh(img)
#smooth out image to remove holes in syllables and threshold out small
#predictions that can't be birds.

  kern = gaussian2d(kernsigma,[kernwidth,kernwidth]) # gaussian convolution kernel
  imgf = conv2(img,kern) #apply convolution
  (m,n) = size(img)

  #loop through and threshold image according to global thresh in config.jl
  for i in 1:m
    for j in 1:n
      if imgf[i,j] < thresh # threshold the image
        imgf[i,j] = 0
      else
        imgf[i,j] = 1
      end
    end
  end
  return imgf
end

function loadpaths()

  # Define all relative paths from config.jl. Changes should only be made
  # to config.jl
  if path_rel
    projdir = pwd()[1:end-3]; #remove src directory
    specpath = string(projdir,specdir)
    labelledpath = string(projdir,labelleddir)
    trainpath = string(projdir,traindir)
    modelpath = string(projdir,modeldir)
  else
    specpath = specdir
    labelledpath = labelleddir
    trainpath = traindir
    modelpath = modeldir
  end
  return specpath,labelledpath,trainpath,modelpath

end


function segmentimg(img)
#=this function takes the predicted image and extracts the nonintersecting
sets of 1's into separate regions. ie. img will be a binary image and segmentimg
will
=#

  #initialise variables
  mask = zeros(size(img))
  (m,n) = size(img)
  labels = []
  eqsets = Any[]
  count = 0

  #this is a somewhat complex algorithm to segment the regions in the image
  #into separate numbered labels. Each region (in nonconvex) may have multiple
  #numbers labelling it but they will be put into equivalence classes and
  #unioned together at the end.
  for c in 1:n
    r = 1
    while r <= m
      if img[r,c] == 1
        top = r
        eq = []
        if isempty(labels)
          count = 1
        else
          count = maximum(labels)+1
        end
        while (r <= m) & (img[r,c] == 1)
          if c > 1
            if img[r,c-1] == 1 #check if syllable was found in prior column
              if mask[r,c-1] != count
                push!(eq,mask[r,c-1]) # add this label to current equiv class
              end
              count = min(mask[r,c-1],count)
            end
          end
          r += 1
        end
        eq = setdiff(eq,1:count)
        if ~isempty(eq)
          eq = [eq;count]
          eqsets = addeq(eqsets,eq) #add eq to the current equivalence class
        end
        #label this region column as count
        mask[top:r-1,c] = count;
        push!(labels,count)
      end
      r += 1
    end
  end
  #union the syllables according to the eqsets
  (mask,area,boxes) = extractsyllables(mask,eqsets)
  return mask,area,boxes
end

function addeq(eqsets,eq)
#add the equivalence pair if it is a new one

  if isempty(eqsets)
    push!(eqsets,eq)
  else
    #look through current eq classes to find count
    found = false
    for i in 1:length(eqsets)
      if ~isempty(intersect(eqsets[i],eq)) # if eq and current set have nontrivial intersection then found class
        eqsets[i] = union(eqsets[i],eq)
        found = true
      end
    end
    if ~found
      push!(eqsets,eq)
    end
  end
  return eqsets
end



function extractsyllables(mask,eqsets)
# This function takes the mask of the image regions and corresponding
# equivalence classes to return the bounding boxes and their areas

  boxes = Any[]
  area = []
  (m,n) = size(mask)

  #useful comprehensions to quickly lookup rows and columns
  rows = [x for x in 1:m, y in 1:n]
  cols = [y for x in 1:m, y in 1:n]

  #union equivalence classes together using the min of each class as the
  #resulting label
  for i in 1:length(eqsets)
    label = minimum(eqsets[i])
    for j in 1:length(eqsets[i])
      mask[find(mask.==eqsets[i][j])] = label
    end
  end

  #reorder the levels in the mask
  levs = setdiff(levels(mask),0)
  current = 1
  for i in 1:length(levs)
    ind = find(mask.==levs[i])
    if length(ind) > minarea
      push!(area,length(ind))
      (r,c) = rows[ind],cols[ind]
      #this may seem the wrong way around but top of image is index 1 which is highest frequency
      top = minimum(r)
      bottom = maximum(r)
      left = minimum(c)
      right = maximum(c)
      push!(boxes,Bounds2(right,top,left,bottom)) # add the bounding box to list
      mask[find(mask.==levs[i])] = i
    else
      mask[find(mask.==levs[i])] = 0
    end
  end
  return mask,area,boxes
end

function dumpsyllables(boxes,modelpath,spec,recorderdata,nrows,ncols)
#this function will take the bounding boxes from one spectrogram and save
# them to file as frequency bounds and time bounds. The time will be absolute in
# seconds, for each different wav file

  tstart,tstop,fmax,fmin,wav = [],[],[],[],[]
  n = 2*boxheight + nrows
  freq = getfreqvector(getrecorder(spec),recorderdata,n)
  fs = freq[1]*2

  #extract each syllable and convert pixels into times and frequencies
  for box in boxes
    top = freq[Int(box.y_max)]
    bottom = freq[Int(box.y_min)]
    (start,stop)=gettime(box,spec,ncols)
    push!(tstart,start),push!(tstop,stop) # push times
    push!(fmax,top),push!(fmin,bottom) #push frequencies
    push!(wav,spec[1:rsearch(spec,"_")[1]]) #push file
  end

  # construct the data into a dataframe
  data = DataFrame(wav=wav,fmin=fmin,fmax=fmax,tstart=tstart,tstop=tstop)

  # append the syllables to file. If file is new then add header otherwise just
  # append the data
  if filesize(string(modelpath,"syllables.csv")) == 0
    writetable(string(modelpath,"syllables.csv"),data,append=true)
  else
    writetable(string(modelpath,"syllables.csv"),data,append=true,header=false)
  end

end

function gettime(box,spec,ncols)

  # get the delta in time between pixels
  dt = speclength/(2*boxwidth + ncols);
  specnum = parse(Int,spec[rsearch(spec,"_")[1]+1:end-4]) #spec number
  tbefore = specnum*speclength # time elapsed before current spectrogram (10 seconds each)
  start = tbefore + dt*(box.x_min + boxwidth)
  stop = tbefore + dt*(box.x_max + boxwidth)
  return start, stop

end

function syllablesfileprompt()
#prompt the user to decide wether syllables.csv should be overwritten or appended to
#if it is nonempty when predict.jl is first run.

  println("\nWARNING: The file data/model/syllables.csv already exists and is nonempty.
Do you wish to append to this file (y/n). If you select (n) the file will be overwritten")
  response = lowercase(readline(STDIN))
  if response[1] == 'y'
    doappend = true
  elseif response[1] == 'n'
    doappend = false
  else
    #wrong response recursively recall this function
    println("Please input a valid response (y/n)")
    doappend = syllablesfileprompt();
  end
  return doappend
end

end
