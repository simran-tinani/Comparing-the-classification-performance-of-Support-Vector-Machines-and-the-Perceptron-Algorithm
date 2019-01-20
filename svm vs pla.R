N <- 100 # number of training points
runs <- 1 # Initializing the number of runs
count <- 0 # Count of number of times Eout for SVM < Eout for PLA (SVM performs better)
sv <- matrix(nrow = 1, ncol = 1000) # number of support vectors in each iteration
sv[,] = 0

while(runs<1001){
  try({
  x <- runif(2, min = -1, max = 1) # Picking a random point in (-1, 1)
  y <- runif(2, min = -1, max = 1)
  fit <- (lm(y~x)) # the line through x and y
  t <- summary(fit)$coefficients[,1] # slope and intercept
  f <- function(x){ # A random line in the plane defined by x and y
    t[2]*x + t[1]
  }
  
  A <- matrix(ncol=N, nrow=2) # Training data set, generated randomly, as required by question
  b <- matrix(ncol=N, nrow=1) # training labels
  
  for(i in 1:N){
    A[, i] <- c(runif(2, min = -1, max = 1))
    b[1, i] <- sign(A[2, i] - f(A[1, i])) # which side of the line the training points lie on
  }
  
  if(length(range(b))==1){next} # Move to the next iteration (run) if all training points lie on the same side of the line
  
  w <- matrix(ncol=1, nrow=3) # Weight vector, to be calculated using PLA
  w[, 1] <- 0
  g <- function(z){
    t(w) %*% z
  }
  
  i <- 0
  while(i < N+1){ # Running the Perceptron Learning Algorithm
    j <- sample(1:N, 1)
    if((sign(g(c(1, A[, j]))) == b[1, j]) == 0){
      w <- w + b[1, j]*c(1, A[, j])
    }
    i = i + 1
  }
  
  S <- matrix(ncol=10000, nrow=2) # Testing data set, random points in XY plane
  for(v in 1:10000){
    S[, v] <- c(runif(2, min = -10000, max = 10000))
  }
  
  # PLA 
  m <- 0 # counter of E_out (PLA)
  v <- 1
  while(v < 10001){
    if(sign(g(c(1, S[, v]))) != sign(S[, v][2] - f(S[, v][1]))){
      m = m + 1
    }
    v = v + 1
  }
  
  # SVM 
  
  require(quadprog)
  
  # Implementing classification using the quadprog package, solve.QP command
  Dmat <- matrix(nrow = N, ncol = N)
  dvec <- matrix(nrow = N, ncol = 1)
  Amat <- matrix(nrow= N+1, ncol = N)
  Amat[,] <- 0
  bvec <- matrix(nrow = N+1, ncol =1)
  bvec[,1]<-0
  for(i in 1:N){
    for(j in 1:N){
      Dmat[i, j] = b[i]*b[j]*t(A[,i])%*%A[,j] 
      dvec[i,1] = 1
      if(i>1 && i==j+1) {
        Amat[i,j] <- 1
      }
      else Amat[i,j] = 0
      bvec[i] = 0
    }
  }
  Dmat <- Dmat + 10^(-8)*diag(N) # Done to solve the problem of non-positive definite matrix
  
  Amat[1,] = t(b)
  Amat[N+1,N]=1  # Amat has its first row as the vector b (as one constraint is b^T*alpha = 0), and the rest as identity because we need all alpha_i's greater than or equal to 0
  Amt = t(Amat)
  
  sol  <- solve.QP(Dmat, dvec, Amt, bvec, meq=1) # meq = 1 because the first inequality must be treated as an equality
  
  l = as.vector(sol["solution"])
  alpha=0
  for(i in 1:N){
    alpha[i]<-as.numeric(rapply(l,c)[i]) # alpha records the obtained parameters
  }
  
  w1 <- 0 # 2-column weight vector (excludes intercept term)
  for(n in 1:N){
    w1 <- w1 + alpha[n]*b[n]*A[,n] # Formula for obtaining weights using alpha's
  }
  
  for(n0 in 1:N){
    if(abs(alpha[n0]) > 10^(-1)){
      break   # Picking an index corresponding to an actual support vector
    }
  }
  for(n in 1:N){
    if(abs(alpha[n]) > 10^(-1) & alpha[n]>0){
      sv[runs] <- sv[runs]+1  # Support vector counter
    }
  }
  
  w0 <- (1-t(w1)%*%A[,n0]*b[n0])/b[n0] # Formula for first entry of final weight vector, corresponding to intercept term
  w2 <- c(w0,w1) # final weight vector
  m1 <- 0 # Eout counter for SVM
  v1 <- 1
  
  g1 <- function(z){ #function: multiplication by the weight vector
    t(w2) %*% z
  }
  
  while(v1 < 10001){ # Counting the number of errors
    if(sign(g1(c(1, S[, v1]))) != sign(S[, v1][2] - f(S[, v1][1]))){
      m1 <- m1 + 1
    }
    v1 <- v1 + 1
  }
  
  if(m>m1){count = count + 1/1000} # If out-of-sample error m in PLA is higher than out-of-sample error in SVM, add to the count 
 } )
  runs = runs+1
}

count*100   # corresponds to the percentage of times SVM performed better:
#code yields about 90% for N =100 and about 80% for N = 10