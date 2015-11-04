# extract_spectrograms.jl
# This script takes audio recordings and dumps the corresponding spectrograms
# into trainpath, defined in the config file.
# author: Timothy Evans
# date: 30/07/14

push!(LOAD_PATH, pwd())
using DSP, WAV, DataFrames, Images
using config, dataprocessing, ProgressMeter

#load paths from config.jl
specpath,labelledpath,trainpath,modelpath = loadpaths()

#= loop through the levels of recording directories. i.e. BSR01/140107/concat/
 NOTE: this is as per the file structure received from Eric Watson with the
 Ark in the Park. If structure is different the following nested loop needs to
 be adapted. =#

fss = Uint[]
p = Progress(length(readdir(specpath)),1) # initialise progress bar

for recorder in readdir(wavpath)
  recdatepath = string(wavpath,recorder,"/")
  for day in readdir(recdatepath)
    recpath = string(recdatepath,day,"/concat/")
    for file in readdir(recpath)
      filepath = string(recpath,file)
      rec,fs = wavread(filepath)
      frec = filterwav(rec,fs)
      #spectrogram segments will be spaced as per defined speclength
      segments = 1:fs*speclength:length(frec)
      specname = string(recorder,"_",day,"_",file[1],"_")
      dumpspectrograms(frec,segments,trainpath,specname,specwidth,p)
    end
  end
  push!(fss,fs)
end

recorderdata = DataFrame(recorders=readdir(wavpath),fsample=fss)
writetable(string(modelpath,"recorderdata.csv"),recorderdata)
