## kokako-detection
Random Forest classifier to identify kokako calls in audio recordings

**Use with Julia**

There are four main files in this package, and four corresponding calls from
within Julia. These will need be called in the following order. From within Julia
you can call these as follows

```
include("extract_spectrograms.jl")
include("extract_features.jl")
include("train.jl")
include("predict.jl")
```
