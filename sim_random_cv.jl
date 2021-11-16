include("spcov.jl")
using RCall
#use same data as R
R"""
library(MASS)
library(spcov)
p <- 20 
n <- 100   #60
model <- "R"    # Choose from "C"liques, "H"ubs, "R"andom and "F"irst-order moving average
n_sim = 30

S_list = array(NA,dim=c(n_sim,p,p))
X_list = array(NA,dim=c(n_sim,n,p))
est_list = array(0,dim=c(n_sim,p,p))


#set.seed(1990)
set.seed(123)
# Build the True Covariance Matrix
Sigma <- matrix(0, nrow = p, ncol = p)
if(model == "C"){
  Sig1 <- matrix(sign(rnorm(n = p^2 / 25)), nrow = p / 5, ncol = p / 5)
  Sig2 <- matrix(sign(rnorm(n = p^2 / 25)), nrow = p / 5, ncol = p / 5)
  Sig3 <- matrix(sign(rnorm(n = p^2 / 25)), nrow = p / 5, ncol = p / 5)
  Sig4 <- matrix(sign(rnorm(n = p^2 / 25)), nrow = p / 5, ncol = p / 5)
  Sig5 <- matrix(sign(rnorm(n = p^2 / 25)), nrow = p / 5, ncol = p / 5)
  Sigma[1:(p / 5), 1:(p / 5)] <- Sig1
  Sigma[(p / 5 + 1):(2 * p / 5), (p / 5 + 1):(2 * p / 5)] <- Sig2
  Sigma[(2 * p / 5 + 1):(3 * p / 5), (2 * p / 5 + 1):(3 * p / 5)] <- Sig3
  Sigma[(3 * p / 5 + 1):(4 * p / 5), (3 * p / 5 + 1):(4 * p / 5)] <- Sig4
  Sigma[(4 * p / 5 + 1):p, (4 * p / 5 + 1):p] <- Sig5
}else if(model == "H"){
  Sigma[1, 1:(p / 5)] <- sign(rnorm(n = p / 5))
  Sigma[(p / 5 + 1), (p / 5 + 1):(2 * p / 5)] <- sign(rnorm(n = p / 5))
  Sigma[(2 * p / 5 + 1), (2 * p / 5 + 1):(3 * p / 5)] <- sign(rnorm(n = p / 5))
  Sigma[(3 * p / 5 + 1), (3 * p / 5 + 1):(4 * p / 5)] <- sign(rnorm(n = p / 5))
  Sigma[(4 * p / 5 + 1), (4 * p / 5 + 1):p] <- sign(rnorm(n = p / 5))
}else if(model == "R"){
  Sigma <- matrix(rbinom(p^2, size = 1, prob = 0.02) * sign(rnorm(n = p^2)), nrow = p, ncol = p)
}else if(model == "F"){
  ind <- matrix(nrow = p - 1, ncol = 2)
  ind[, 1] <- 1:(p - 1)
  ind[, 2] <- 2:p
  Sigma[ind] <- 0.4
  ind[, 1] <- 2:p
  ind[, 2] <- 1:(p - 1)
  Sigma[ind] <- 0.4
}else{
  print("Specify valid covariance model")
}


diag(Sigma) <- 0
ind <- lower.tri(Sigma, diag = FALSE)
Sigma[ind] <- t(Sigma)[ind] # symmetrize
dd <- (eigen(Sigma)$values[1] - eigen(Sigma)$values[p] * p) / (p - 1)
diag(Sigma) <- dd   # condition number is now p

for(i in 1:n_sim){
  # generate the training data
  X <- mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  X_list[i,,] <- X

  S <- matrix(0, nrow = p, ncol = p)
  for(j in 1:n){
       S <- S + X[j, ] %*% t(X[j, ])
  }
  S <- S / n + diag(x = 10^(-2), nrow = p)
  #S = cov(X) + diag(x = 10^(-2), nrow = p) #conditioning
  S_list[i,,] <- S
}
"""

@rget n_sim S_list X_list Sigma p n est_list

num_k = 15
tru = Int((count(abs.(Sigma).>0.0000001) - p)/2 )
k_list = round.(Int, range( .9*tru,stop= 1.15*tru,length=num_k ) )
push!(k_list,tru)
k_list = sort(unique(k_list))


ρ = 0.3 #0.1 #initial value
maxiter = 200
#tol = 1e-5
nFolds = 5
tol = 1e-7


Zero = zeros(n_sim);
rmse = zeros(n_sim);
entropyloss = zeros(n_sim);
MinEv = zeros(n_sim);
TP = zeros(n_sim);
FP = zeros(n_sim);
TN = zeros(n_sim);
FN = zeros(n_sim);
times = zeros(n_sim);
cv_k = zeros(n_sim);

for i=1:Int(n_sim) 
    println("================================ Repeat sim number $i =================================") 
        S = S_list[i,:,:];
        X = X_list[i,:,:];
        param_best = crossval(X,nFolds,k_list,tol,maxiter,ρ);
        times[i] = @elapsed est = project_sparse_symmetric( sparse_cov(S, ρ, param_best, maxiter,tol), param_best ) 
        est_list[i,:,:] = est;
        Zero[i] = count(x->x==0,est);
        rmse[i] = sqrt(mean((Sigma - est)^2));
        entropyloss[i] = -log(det(est * inv(Sigma))) + sum(diag(est * inv(Sigma))) - p;
        MinEv[i] = eigmin(est);
        TP[i] = length(intersect(findall(est.!=0), findall(Sigma.!=0)));
        FP[i] = length(intersect(findall(est.!=0), findall(Sigma.==0)));
        TN[i] = length(intersect(findall(est.==0), findall(Sigma.==0)));
        FN[i] = length(intersect(findall(est.==0), findall(Sigma.!=0))) ;
        cv_k[i] = param_best;
end

using JLD
@save "cv100_2%.jld"
