using NeuralEstimators
using Flux: loadmodel!
using BSON

import Flux: Bilinear
function (b::Bilinear)(Z::A) where A <: AbstractArray{T, 3} where T
    @assert size(Z, 2) == 2
    x = Z[:, 1, :]
    y = Z[:, 2, :]
    b(x, y)
end

# NBEs - Parameter estimation

## Model W
d = 2; 
p = 2;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, softplus), 
  Dense(w, 1, identity) 
)

Ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q + 1, w, relu), final_layer)
architecture = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(architecture)
q̂ = IntervalEstimator(architecture)

BSON.@load "NBE/Random Scale Construction/Model W/best_network.bson" model_state;
θ̂_w = deepcopy(θ̂);
loadmodel!(θ̂_w, model_state);

BSON.@load "NBE/Interval Estimator/Model W/best_network.bson" model_state;
q̂_w = deepcopy(q̂);
loadmodel!(q̂_w, model_state);

## Model HW
d = 2; 
p = 2;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Chain(Dense(w, 1), Compress(0, 1)),
  Chain(Dense(w, 1), Compress(-1, 1))
)

Ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q + 1, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/Random Scale Construction/Model HW/best_network.bson" model_state;
θ̂_hw = deepcopy(θ̂);
loadmodel!(θ̂_hw, model_state);

## Model E1
d = 2; 
p = 3;
w = 128;
q = 256;

Ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q + 1, w, relu), Dense(w, p, softplus))
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/Random Scale Construction/Model E1/best_network.bson" model_state;
θ̂_e1 = deepcopy(θ̂);
loadmodel!(θ̂_e1, model_state);

## Model E2
d = 2; 
p = 2;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, softplus), 
  Dense(w, 1, identity) 
)

Ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q + 1, w, relu), final_layer)
architecture = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(architecture)

BSON.@load "NBE/Random Scale Construction/Model E2/best_network.bson" model_state;
θ̂_e2 = deepcopy(θ̂);
loadmodel!(θ̂_e2, model_state);

## WCM with Gaussian and Logistic
d = 2;
p = 3;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, identity),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/WCM/Gaussian and Logistic/best_network.bson" model_state
θ̂_wcm1 = deepcopy(θ̂);
loadmodel!(θ̂_wcm1, model_state);

## WCM with Frank and Joe
d = 2;
p = 3;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Chain(Dense(w, 1), Compress(1, 15)),
  Dense(w, 1, identity),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/WCM/Frank and Joe/best_network.bson" model_state
θ̂_wcm2 = deepcopy(θ̂);
loadmodel!(θ̂_wcm2, model_state);

## WCM with Gaussian and W
d = 2;
p = 4;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, softplus),
  Dense(w, 1, identity),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/WCM/Gaussian and W/best_network.bson" model_state
θ̂_wcm_w = deepcopy(θ̂);
loadmodel!(θ̂_wcm_w, model_state);

## WCM with Gaussian and HW
d = 2;
p = 4;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Chain(Dense(w, 1), Compress(0, 1)),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/WCM/Gaussian and HW/best_network.bson" model_state
θ̂_wcm_hw = deepcopy(θ̂);
loadmodel!(θ̂_wcm_hw, model_state);

## WCM with Gaussian and E1
d = 2;
p = 4;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, softplus),
  Dense(w, 1, softplus),
  Dense(w, 1, softplus),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)
q̂ = IntervalEstimator(deepset)

BSON.@load "NBE/WCM/Gaussian and E1/best_network.bson" model_state
θ̂_wcm_e1 = deepcopy(θ̂);
loadmodel!(θ̂_wcm_e1, model_state);

BSON.@load "NBE/Interval Estimator/WCM with Gaussian and E1/best_network.bson" model_state;
q̂_wcm_e1 = deepcopy(q̂);
loadmodel!(q̂_wcm_e1, model_state);

## WCM with Gaussian and E1
d = 2;
p = 4;
w = 128;
q = 256;

final_layer = Parallel(
  vcat,
  Dense(w, 1, softplus),
  Dense(w, 1, identity),
  Chain(Dense(w, 1), Compress(-1, 1)),
  Dense(w, 1, identity)
)

Ψ = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, q, relu))
Φ = Chain(Dense(q, w, relu), final_layer)
deepset = DeepSet(Ψ, Φ)
θ̂ = PointEstimator(deepset)

BSON.@load "NBE/WCM/Gaussian and E2/best_network.bson" model_state
θ̂_wcm_e1 = deepcopy(θ̂);
loadmodel!(θ̂_wcm_e1, model_state);

# Neural classifier - Model selection

## Binary Classification

### Models W and HW
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
θ̂  = DeepSet(ψ, ϕ);

BSON.@load "Neural Classifier/Binary Classification/Models W and HW/best_network.bson" model_state
θ̂_whw = deepcopy(θ̂);
loadmodel!(θ̂_whw, model_state);

## Models W and E1
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂  = PointEstimator(deepset)

BSON.@load "Neural Classifier/Binary Classification/Models W and E1/best_network.bson" model_state
θ̂_we1 = deepcopy(θ̂);
loadmodel!(θ̂_we1, model_state);

## Models W and E2
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂  = PointEstimator(deepset)

BSON.@load "Neural Classifier/Binary Classification/Models W and E2/best_network.bson" model_state
θ̂_we2 = deepcopy(θ̂);
loadmodel!(θ̂_we2, model_state);

## Models HW and E1
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂  = PointEstimator(deepset)

BSON.@load "Neural Classifier/Binary Classification/Models HW and E1/best_network.bson" model_state
θ̂_hwe1 = deepcopy(θ̂);
loadmodel!(θ̂_hwe1, model_state);

## Models HW and E2
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂  = PointEstimator(deepset)

BSON.@load "Neural Classifier/Binary Classification/Models HW and E2/best_network.bson" model_state
θ̂_hwe2 = deepcopy(θ̂);
loadmodel!(θ̂_hwe2, model_state);

## Models E1 and E2
d = 2;
p = 2;
w = 128;
q = 128;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂  = PointEstimator(deepset)

BSON.@load "Neural Classifier/Binary Classification/Models E1 and E2/best_network.bson" model_state
θ̂_e1e2 = deepcopy(θ̂);
loadmodel!(θ̂_e1e2, model_state);

## Multiclass Classification
d = 2;
p = 4;
w = 128;
q = 256;

ψ = Chain(Bilinear((d, d) => w, relu), Dense(w, w, relu), Dense(w, q, relu));
ϕ = Chain(Dense(q + 1, w, relu), Dense(w, p), softmax);
deepset = DeepSet(ψ, ϕ);
θ̂ = PointEstimator(deepset)

BSON.@load "Neural Classifier/Multiclass Classification/best_network.bson" model_state
θ̂_ms = deepcopy(θ̂);
loadmodel!(θ̂_ms, model_state);
