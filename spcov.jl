using Statistics; using LinearAlgebra

function cov2cor(C::AbstractMatrix)
   #should check if C is positive semidefinite
   sigma = sqrt.(diag(C))
   return C ./ (sigma*sigma')
end


function project_sparse_symmetric(
  Y::AbstractMatrix{T},
  k::Int) where T <: Union{Float32, Float64}
# This function projects the square matrix Y onto the set 
# of symmetric matrices with at most k upper triangular 
# nonzero entries.
   n = size(Y,1)
   m = Int(n*(n-1)/2)

# Record the upper triangular entries in a vector u.
   u = zeros(m)
   l = 0
   for i = 1:n-1
      for j = i+1:n
         l = l+1
         u[l] = Y[i,j]
      end
   end

# Sort the entries of u and zero out all but k. 
   perm = sortperm(abs.(u))
   for i = 1:m-k
      u[perm[i]] = 0.0   
   end

# Output the results in the symmetric matrix Z.
   Z = zeros(size(Y))
   l = 0
   for i = 1:n
      Z[i,i] = Y[i,i]
      for j = i+1:n
         l = l+1
         Z[i,j] = u[l]
         Z[j,i] = u[l]
      end
   end    
   return Z
end

function project_sparse_corr(
  Y::AbstractMatrix{T},
  k::Int) where T <: Union{Float32, Float64}
# This function projects the square matrix Y onto the set 
# of symmetric matrices with at most k upper triangular 
# nonzero entries.
   n = size(Y,1)
   m = Int(n*(n-1)/2)

# Record the upper triangular entries in a vector u.
   u = zeros(m)
   l = 0
   for i = 1:n-1
      for j = i+1:n
         l = l+1
         u[l] = Y[i,j]
      end
   end

# Sort the entries of u and zero out all but k. 
   perm = sortperm(abs.(u))
   for i = 1:m-k
      u[perm[i]] = 0.0   
   end

# Output the results in the symmetric matrix Z.
   Z = zeros(size(Y))
   l = 0
   for i = 1:n
      Z[i,i] = Y[i,i]
      for j = i+1:n
         l = l+1
         Z[i,j] = u[l]
         Z[j,i] = u[l]
      end
   end
   #Z[1:(n+1):end] = 1   # diagonal constraint for correlation
   #return Z
   return cov2cor(Z)
end

function sparse_cov(
  S::AbstractMatrix{T},
  ρ::T,
  k::Int,
  maxiters::Int = 1000,
  funtol::T = T(1e-6),
  ) where T <: Union{Float32, Float64}

  n = size(S)[1]
  if k == n*(n-1)/2
    println("Trivial Case")
    return project_sparse_symmetric(S,n^2)
  end

  Σ = project_sparse_symmetric(S,0) #initialize as diagonal
  
  if k == n*(n-1)/2
    println("Trivial Case")
    return Σ
  end

  Σ_proj = project_sparse_symmetric(Σ,k)
  #Σ_inv = inv(Σ)
  Σ_new = similar(S)
  #D = similar(S)
  X = similar(S)
  #ρ_update = 5 #number of iterations until scaling up rho (p); doubles itself each time
  ρ_increase = T(1.2) #factor by which to scale up rho when updating
  obj = T(999.9) #initialize the objective
  total_backtracks = 0

  for iter in 1:maxiters
    if iter == maxiters
      println("Max iters reached")
    end
    #if mod(iter,ρ_update)==0 #schedule to scale up penalty parameter
    if ρ < 1e6
      ρ *= ρ_increase
    end

    # update used to be X = sylvester(ρ*Σ, Σ_inv, -Σ*D), but better memory management now
    #D = (ρ*Σ_proj + Σ_inv*S*Σ_inv)
    X = sylvester(ρ*Σ, inv(Σ), -Σ*(ρ*Σ_proj + inv(Σ)*S*inv(Σ))) #minimize by solving a Sylvester equation
    #check against closed form
    #X = reshape( inv( ρ*eye(n^2) + kron(Σ_inv,Σ_inv) )*vec(D), size(Σ) ) 
    #show(vecnorm(X-closed))    
    Σ_new = X #this is temporarily stored for backtracking
    j = 1
    #while !isposdef(Σ_new)
    while minimum(real(eigvals(Σ_new)[1])) < 0 
      #println( eigvals(Σ_new)[1] )
      Σ_new = Σ + 1/(2^j)*(X - Σ)
      j+=1
      total_backtracks += 1
      #show(isposdef(Σ_new))
    end
    #if j > 1
    #  println("Number of backtracks: $(j-1)" )
    #end
    if j > 14
      println("More than 15 backtracks at iter $iter")
    end

    if j > 20
      println("Probably boundary solution at iter $iter: stopping")
      return Σ
    end
    Σ = Σ_new
    #Σ_inv = inv(Σ) #this is the inverse to be used in the NEXT iteration!
    Σ_proj = project_sparse_symmetric(Σ,k)

    #objective for stopping criterion
    objold = obj
    obj = logdet(Σ) + tr(inv(Σ)*S)# + ρ*vecnorm(Σ-Σ_proj)/2
    @simd for i in 1:n^2
      obj += ρ*abs2(Σ[i] - Σ_proj[i])/2
    end

    #stopping
    if iter > 4 && abs(obj - objold) < funtol * (objold)
        ab = total_backtracks/iter
        println("$iter iterations")
        println("Average backtracks per iter: $ab")
        break
    end

  end
  return Σ
end


function sparse_corr(
  S::AbstractMatrix{T},
  ρ::T,
  k::Int,
  maxiters::Int = 1000,
  funtol::T = T(1e-6)
  ) where T <: Union{Float32, Float64}

  n = size(S)[1]
  Σ = project_sparse_symmetric(S,0) #initialize as symmetrized covariance

  Σ_proj = project_sparse_symmetric(Σ,k)
  Σ_inv = inv(Σ)
  Σ_new = similar(S)
  D = similar(S)
  X = similar(S)
  #ρ_update = 5 #number of iterations until scaling up rho (p); doubles itself each time
  ρ_increase = T(1.2) #factor by which to scale up rho when updating
  obj = T(999.9) #initialize the objective

  for iter in 1:maxiters
    if iter == maxiters
      println("Max iters reached")
    end
    #if mod(iter,ρ_update)==0 #schedule to scale up penalty parameter
    if ρ < 1e14
      ρ *= ρ_increase
    end
    #end

    #Σ_inv = inv(Σ)  # the inverse is now computed AFTER the update in advance of next iter
    D = (ρ*Σ_proj + Σ_inv*S*Σ_inv)
    X = sylvester(ρ*Σ, Σ_inv, -Σ*D) #minimize by solving a Sylvester equation
    #check against closed form
    #X = reshape( inv( ρ*eye(n^2) + kron(Σ_inv,Σ_inv) )*vec(D), size(Σ) ) 
    #show(vecnorm(X-closed))    
    Σ_new = X #this is temporarily stored for backtracking
    j = 1
    #while !isposdef(Σ_new)
    while minimum(real(eigvals(Σ_new)[1])) < 0 
      #println( eigvals(Σ_new)[1] )
      Σ_new = Σ + 1/(2^j)*(X - Σ)
      j+=1
      #show(isposdef(Σ_new))
    end
    if j > 1s
      println("Number of backtracks: $(j-1)" )
    end
    Σ = Σ_new
    Σ_inv = inv(Σ) #this is the inverse to be used in the NEXT iteration!
    Σ_proj = project_sparse_corr(Σ,k)

    #objective for stopping criterion
    objold = obj
    obj = logdet(Σ) + tr(Σ_inv*S)# + ρ*vecnorm(Σ-Σ_proj)/2
    @simd for i in 1:n^2
      obj += ρ*abs2(Σ[i] - Σ_proj[i])/2
    end

    #stopping
    if iter > 4 && abs(obj - objold) < funtol * (objold)
        println("$iter iterations")
        break
    end

  end
  return Σ
end

function crossval(
    X::AbstractMatrix{T},
    nFolds::Int,
    param_list::Array{Int},
    funtol::T=T(1e-6),
    maxiters::Int=500,
    ρ = T(0.001),
    seed::Int=12345,
    ) where T <: Union{Float32, Float64}

    #srand(seed)
    nParams = length(param_list)
    n, d = size(X)
    testSize = Int(n/nFolds)
    folds = repeat( collect(1:nFolds), inner=[testSize])
    cv_loglike = zeros(nFolds)
    #initialize
    loglike_list = zeros(nParams)
    loglike_sd_list = zeros(nParams)

    for i in 1:nParams
        param = param_list[i]
        println("Sparsity k: $param")

        for l = 1:nFolds
            print("Fold $l:   ")
            testInds = findall(folds.==l)
            trainInds = findall(folds.!=l)
            X_train = X[trainInds,:]
            X_test = X[testInds,:]
            n_train = length(trainInds)
            
            if d>(n-1)
              estimate = sparse_cov(cov(X_train)+.001*I, ρ, param, maxiters, funtol) 
            else 
              estimate = sparse_cov(cov(X_train), ρ, param, maxiters, funtol)
            end


            #evaluate loglikelihood on test data
            cv_loglike[l] = logdet(estimate) + tr(inv(estimate)*cov(X_test))
        end

        loglike_list[i] = mean(cv_loglike)
        loglike_sd_list[i] = std(cv_loglike)
    end

    param_best = param_list[ argmin(loglike_list) ]
    #loglike_best = minimum(loglike_list)
    return param_best #,  loglike_best, loglike_list, loglike_sd_list
end


# to be used with the international migration data; accounts for overlapping folds to avoid trivial standard deviations
function crossval_corr(
    X::AbstractMatrix{T},
    param_list::Array{Int},
    funtol::T=T(1e-6),
    nFolds::Int=5,
    maxiters::Int=500,
    seed::Int=12345,
    ρ = T(0.001)
    ) where T <: Union{Float32, Float64}

    #srand(seed)
    nParams = length(param_list)
    n, d = size(X)
    testSize = Int((n-2)/nFolds)
    folds = repeat( collect(1:nFolds), inner=[testSize])
    cv_loglike = zeros(nFolds)
    #initialize
    loglike_list = zeros(nParams)
    loglike_sd_list = zeros(nParams)

    for i in 1:nParams
        param = param_list[i]
        println("Sparsity k: $param")

        for l = 1:nFolds
            print("Fold: $l")
            testInds = [1:2; findall(folds.==l) + 2]
            trainInds = [1:2; findall(folds.!=l) + 2]
            X_train = X[trainInds,:]
            X_test = X[testInds,:]
            n_train = length(trainInds)
            cor_train = .999*cor(X_train) + .001*I #for pos def
            cor_test = .999*cor(X_test) + .001*I
            estimate = sparse_corr(cor_train, ρ, param, maxiters) 
            #evaluate the negative loglikelihood on test data
            cv_loglike[l] = logdet(estimate) + tr(inv(estimate)*cor_test)
        end

        loglike_list[i] = mean(cv_loglike)
        loglike_sd_list[i] = std(cv_loglike)
    end

    param_best = param_list[ argmin(loglike_list) ]
    loglike_best = minimum(loglike_list)
    return param_best, loglike_best, loglike_list, loglike_sd_list
end

#returns mean negative log-likelihood on test set using same folds as crossval, given an estimate of Σ
function mean_test_ll(
    X::AbstractMatrix{T},
    estimate::AbstractMatrix{T},
    nFolds::Int=3,
    seed::Int=12345,
    ) where T <: Union{Float32, Float64}

    #srand(seed)
    n, d = size(X)
    testSize = Int((n-2)/nFolds)
    folds = repeat( collect(1:nFolds), inner=[testSize])
    cv_loglike = zeros(nFolds)

    for l = 1:nFolds
        testInds = [1:2; findall(folds.==l) + 2]
        trainInds = [1:2; findall(folds.!=l) + 2]
        X_test = X[testInds,:]
        cor_test = .99*cor(X_test) + .01*I
        #evaluate the negative loglikelihood on test data
        cv_loglike[l] = logdet(estimate) + tr(inv(estimate)*cor_test)
    end

    return mean(cv_loglike), std(cv_loglike)
end


