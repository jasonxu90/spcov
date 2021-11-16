library(MASS)
library(spcov)
library(CVTuningCov)
library(PDSCE)
p <- 20
n <- 100   #60
model <- "R"    # Choose from "C"liques, "H"ubs, "R"andom and "F"irst-order moving average
n_sim = 40
num_lambda <- 40 #15 
tol <- 1e-6 # 0.1
t <- 10 

S_list = array(NA,dim=c(n_sim,p,p))
X_list = array(NA,dim=c(n_sim,n,p))

set.seed(1990)
#set.seed(123)
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

nFold = 5
k.grid <- seq(.2,1,length.out=num_lambda)^2 #better for p=200
#k.grid <- seq(.6,2,length.out=num_lambda)^2 #better for p=100

rmse_hard <- entropyloss_hard <- MinEv_hard <- TP_hard <- FP_hard <- TN_hard <- FN_hard <- rep(NA,n_sim)
rmse_soft <- entropyloss_soft <- MinEv_soft <- TP_soft <- FP_soft <- TN_soft <- FN_soft <- rep(NA,n_sim)
rmse_roth <- entropyloss_roth <- MinEv_roth <- TP_roth <- FP_roth <- TN_roth <- FN_roth <- rep(NA,n_sim)


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

for(i in 1:n_sim){
  X <- X_list[i,,]
  S <- S_list[i,,]
  #print(X[1:3,1:3])
  hard_CV <- regular.CV(X,k.grid, method='HardThresholding',fold=nFold,norm='F', seed=1990);
  soft_CV <- regular.CV(X,k.grid, method='SoftThresholding',fold=nFold,norm='F', seed=1990);
  roth_CV <- pdsoft.cv(X,lam.vec=k.grid,standard=TRUE,init=c("diag"),tau=1e-04,nsplits=nFold)#,n.tr = NULL, tolin = 1e-08, tolout = 1e-08, maxitin = 10000, maxitout = 1000, quiet = TRUE);
    
  print(hard_CV$CV.k[1])
  print(soft_CV$CV.k[1])
  print(roth_CV$best.lam)
  
  est_hard = hard.thresholding(S,hard_CV$CV.k[1])
  est_soft = soft.thresholding(S,soft_CV$CV.k[1])
  est_roth = roth_CV$sigma
  
  rmse_hard[i] <- sqrt(sum((Sigma - est_hard)^2)) / p
  entropyloss_hard[i] <- -log(det(est_hard %*% solve(Sigma))) + sum(diag(est_hard %*% solve(Sigma))) - p
  MinEv_hard[i] <- eigen(est_hard)$values[p]
  TP_hard[i] <- sum(est_hard != 0 & Sigma != 0)
  FP_hard[i] <- sum(est_hard != 0 & Sigma == 0)
  TN_hard[i] <- sum(est_hard == 0 & Sigma == 0)
  FN_hard[i] <- sum(est_hard == 0 & Sigma != 0)
  
  rmse_soft[i] <- sqrt(sum((Sigma - est_soft)^2)) / p
  entropyloss_soft[i] <- -log(det(est_soft %*% solve(Sigma))) + sum(diag(est_soft %*% solve(Sigma))) - p
  MinEv_soft[i] <- eigen(est_soft)$values[p]
  TP_soft[i] <- sum(est_soft != 0 & Sigma != 0)
  FP_soft[i] <- sum(est_soft != 0 & Sigma == 0)
  TN_soft[i] <- sum(est_soft == 0 & Sigma == 0)
  FN_soft[i] <- sum(est_soft == 0 & Sigma != 0)
  
  
  rmse_roth[i] <- sqrt(sum((Sigma - est_roth)^2)) / p
  entropyloss_roth[i] <- -log(det(est_roth %*% solve(Sigma))) + sum(diag(est_roth %*% solve(Sigma))) - p
  MinEv_roth[i] <- eigen(est_roth)$values[p]
  TP_roth[i] <- sum(est_roth != 0 & Sigma != 0)
  FP_roth[i] <- sum(est_roth != 0 & Sigma == 0)
  TN_roth[i] <- sum(est_roth == 0 & Sigma == 0)
  FN_roth[i] <- sum(est_roth == 0 & Sigma != 0)
  
  
  
  # P1 <- matrix(1, nrow = p, ncol = p)	
  # diag(P1) <- 0
  # 
  # time_spcov[i] <- system.time( 
  #   for(j in 1:num_lambda){
  #     est_sparsecov <- spcov(Sigma=S, S=S, lambda=lambda1[j] * P1, step.size=t, n.inner.steps=10, n.outer.steps=5000, trace=0,tol.outer=tol)$Sigma
  #     Zero_sparsecov[i, j] <- sum(est_sparsecov == 0)
  #     rmse_sparsecov[i, j] <- sqrt(sum((Sigma - est_sparsecov)^2)) / p
  #     entropyloss_sparsecov[i, j] <- -log(det(est_sparsecov %*% solve(Sigma))) + sum(diag(est_sparsecov %*% solve(Sigma))) - p
  #     MinEv_sparsecov[i, j] <- eigen(est_sparsecov)$values[p]
  #     TP_sparsecov[i, j] <- sum(est_sparsecov != 0 & Sigma != 0)
  #     FP_sparsecov[i, j] <- sum(est_sparsecov != 0 & Sigma == 0)
  #     TN_sparsecov[i, j] <- sum(est_sparsecov == 0 & Sigma == 0)
  #     FN_sparsecov[i, j] <- sum(est_sparsecov == 0 & Sigma != 0)
  #   } )[3]
  # 
}

mean(rmse_soft)
sd(rmse_soft)
mean(entropyloss_soft)
sd(entropyloss_soft)
mean(1 - TN_soft/(TN_soft + FP_soft))*100
mean(FN_soft/(TP_soft + FN_soft) )*100


mean(rmse_hard,na.rm=T)
sd(rmse_hard,na.rm=T)
entropyloss_hard[entropyloss_hard==Inf] = NA
mean(entropyloss_hard,na.rm=T)
sd(entropyloss_hard,na.rm=T)
mean(1 - TN_hard/(TN_hard + FP_hard))*100
mean(FN_hard/(TP_hard + FN_hard) )*100


mean(rmse_roth)
sd(rmse_roth)
mean(entropyloss_roth)
sd(entropyloss_roth)
mean(1 - TN_roth/(TN_roth + FP_roth))*100
mean(FN_roth/(TP_roth + FN_roth) )*100



#plot the runtimes
pdf("runtime.pdf",height=3.5,width=5.5)
x = c(200, 500, 1000, 2000, 3000, 5000)
y = c(2.35,7.41,58.1,639.4,2128,8401)
barplot(y, main = "", names.arg = x, xlab = "Dimension (p)", ylab = "Runtime (seconds)")
dev.off()


