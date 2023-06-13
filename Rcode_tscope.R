### tscope R code for simulation study ###
## created June 2023

# This is R code for the 2sCOPE method in our paper 'Addressing Endogeneity using 
# a Two-stage Copula Generated Regressor Approach'. 
# Please refer to the function of 2sCOPE method (tscope) in PART 1, 
# and refer to an example of how to use this r function in PART 2

# PLEASE CITE AS:
# Yang, F., Qian, Y., & Xie, H. (2022). Addressing endogeneity using a two-stage 
# copula generated regressor approach (No. w29708). National Bureau of Economic Research.

# Load required libraries below IF YOU HAVE NOT ALREADY.
# SEE https://www.r-bloggers.com/how-to-install-packages-on-r-screenshots/
# FOR INSTRUCTIONS HOW TO INSTALL A PACKAGE

library(neuralnet)
library(MASS)
library(glmnet)
library(nlme)
library(EnvStats)
library(pls)
library(nortest)
library(GoFKernel)
library(mvtnorm)
library(rms)
library(data.table)
library(multiColl)
library(olsrr)
library(aod)
library(scales)


## PART 1: R function of 2sCOPE
rm(list = ls())
tscope <- function(formula, data){
  F.formula <- as.Formula(formula)
  ymodel = formula(F.formula, lhs = 1, rhs = 1)
  x <- as.matrix(model.matrix(F.formula, data = data, rhs = 1)) #all regressors including intercept
  y <- model.part(F.formula, data = data, lhs = 1)[, 1] #dependent variable
  n = nrow(x)
  if (ncol(x) <= 1)  print("No predictor variables specified for the outcome")
  
  #obtain endogenous regressors and exogenous regressors
  endox <- as.matrix(model.matrix(F.formula, data = data, rhs = length(F.formula)[2]))
  endox <- endox[,colnames(endox)!= "(Intercept)", drop=F] ## remove the intercept to obtain endogenous regressors only.
  nendox <- ncol(endox)
  endoxstar <- matrix(0,n,nendox)
  w <- x[ , -which(colnames(x) %in% c("(Intercept)",colnames(endox))), drop=F]
  nw = ncol(w)
  wstar <- matrix(0,n,nw)
  
    #calculate xstar
    for (i in 1:nendox){
      endoxstartemp = ecdf(endox[,i])(endox[,i])
      endoxstartemp[endoxstartemp==1]=n/(n+1) 
      endoxstartemp= qnorm(endoxstartemp)
      endoxstar[,i] = copy(endoxstartemp)
    }
  
    if (nendox >= ncol(x)-1){
      #print("No exogenous regressors specified for the outcome, and use copula_origin directly")
      res <- lm(y ~ -1 + x + endoxstar)
    }else{
      #calculate wstar
      for (i in 1:nw){
        wstartemp = ecdf(w[,i])(w[,i])
        wstartemp[wstartemp==1]=n/(n+1) 
        wstartemp= qnorm(wstartemp)
        wstar[,i] = copy(wstartemp)
      } 
      stage1_resid = matrix(0,n,nendox)
      for (j in 1:nendox){
        stage1_resid[,j]<- lm(endoxstar[,j]~wstar)$resid
      }
      res <- lm(y ~ -1 + x + stage1_resid)
    }
  
    ##endogeneity level: cor(xstar, error)
     resid2=y-c(x%*%res$coef[1:ncol(x)])
     corr_xerror_tscope = rep(0,nendox)
    for (j in 1:nendox){
      corr_xerror_tscope[j]<- cor(endoxstar[,j],resid2)
    }
 
  # print output
  cat("========================================================",fill=TRUE)
  cat("2sCOPE estimates",fill=TRUE)
  print(summary(res))
  res.tab = matrix(0,1,ncol(x)+nendox+1)
  res.tab[1,] = c(t(res$coef[1:(ncol(x))]),t(corr_xerror_tscope),sd(resid2))
  
  return(res.tab)
} 


## PART 2: Example of how to use the function above
nsim = 1000 #repeat 1000 times
nendox = 1 # number of endogenous variables
nw = 1 # number of exogenous variables
tscope.res = matrix(0,nsim,2*nendox+nw+2)
for (m in 1:nsim){
  print(m)
  
  #simulation data: (if you use your own dataset, directly put your data in line 120)
  n = 1000
  rpe = 0.5
  rpw = 0.5
  temp <- rmvnorm(n, mean=c(0,0,0), sigma=cbind(c(1,rpw,rpe),
                                                c(rpw,1,0),
                                                c(rpe,0,1)))

  #p=qnormMix(pnorm(temp[,1]), mean1 = -1, sd1 = 1, mean2 = 1, sd2 = 1, p.mix = 0.5) ## mixture normal
  p = qt(pnorm(temp[,1]), df=10, ncp=0, lower.tail = TRUE, log.p = FALSE)
  w=qexp(pnorm(temp[,2]))
  y <- 1 + p - w + temp[,3]

  data = as.data.frame(cbind(y,p,w)) 
  formula = 'y~p+w|p' #p is endogeneous variable
  res <- tscope(formula=formula, data=data)
  tscope.res[m,] = c(res)
}

## report average estimates and sd across 1000 datasets
tscope.tab = matrix(0,2,ncol(tscope.res))
colnames(tscope.tab) = c('Intercept','P','W','Rho','sdError')
tscope.tab[1,] = apply(tscope.res,2,mean) #average estimates across 1000 data sets
tscope.tab[2,] = apply(tscope.res,2,sd) #average sd across 1000 data sets

tscope.tab

