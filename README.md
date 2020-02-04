# Bayesian
A collection of projects using Bayesian statistics

## 1. Markov Chain Monte Carlo (MCMC)

The fundamental objective of Bayesian data analysis is to determine the posterior distribution

`p(θ | X)=p(X | θ)p(θ)p(X)` 

where the denominator is

`p(X)=∫dθ∗p(X | θ∗)p(θ∗)`

Here,

`p(X | θ)` is the likelihood, `p(θ)` is the prior and `p(X)` is a normalizing constant also known as the evidence or marginal likelihood

We will use the toy example of estimating the bias of a coin given a sample consisting of n tosses to illustrate a few of the approaches.

### Analytical solution 

If we use a beta distribution as the prior, then the posterior distribution has a closed form solution. This is shown in the example below. Some general points:

We need to choose a prior distribtuiton family (i.e. the beta here) as well as its parameters (here a=10, b=10) The prior distribution may be relatively uninformative (i.e. more flat) or inforamtive (i.e. more peaked) The posterior depends on both the prior and the data As the amount of data becomes large, the posterior approximates the MLE An informative prior takes more data to shift than an uninformative one Of course, it is also important the model used (i.e. the likelihood) is appropriate for the fitting the data The mode of the posterior distribution is known as the maximum a posteriori (MAP) estimate (MLE which is the mode of the likelihood).
