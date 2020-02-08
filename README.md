# Bayesian Deep Learning
A collection of projects using Bayesian deep learning

## If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v:

## 1. Markov Chain Monte Carlo (MCMC)

The fundamental objective of Bayesian data analysis is to determine the posterior distribution

`p(θ | X)=p(X | θ)p(θ)p(X)` 

where the denominator is

`p(X)=∫dθ∗p(X | θ∗)p(θ∗)`

Here,

`p(X | θ)` is the likelihood, `p(θ)` is the prior and `p(X)` is a normalizing constant also known as the evidence or marginal likelihood

We will use the toy example of estimating the bias of a coin given a sample consisting of n tosses to illustrate a few of the approaches.

Analytical solution 

If we use a beta distribution as the prior, then the posterior distribution has a closed form solution. This is shown in the example below. Some general points:

We need to choose a prior distribtuiton family (i.e. the beta here) as well as its parameters (here a=10, b=10) The prior distribution may be relatively uninformative (i.e. more flat) or inforamtive (i.e. more peaked) The posterior depends on both the prior and the data As the amount of data becomes large, the posterior approximates the MLE An informative prior takes more data to shift than an uninformative one Of course, it is also important the model used (i.e. the likelihood) is appropriate for the fitting the data The mode of the posterior distribution is known as the maximum a posteriori (MAP) estimate (MLE which is the mode of the likelihood).


![](images/mcmc1.png)

![](images/mcmc2.png)

## 2. Variational Autoencoder (VAE)

Variational Autoencoder (VAE): in neural net language, a VAE consists of an encoder, a decoder, and a loss function. In probability model terms, the variational autoencoder refers to approximate inference in a latent Gaussian model where the approximate posterior and model likelihood are parametrized by neural nets (the inference and generative networks).

Loss function: in neural net language, we think of loss functions. Training means minimizing these loss functions. But in variational inference, we maximize the ELBO (which is not a loss function). This leads to awkwardness like calling optimizer.minimize(-elbo) as optimizers in neural net frameworks only support minimization.

Encoder: in the neural net world, the encoder is a neural network that outputs a representation zz of data xx. In probability model terms, the inference network parametrizes the approximate posterior of the latent variables zz. The inference network outputs parameters to the distribution q(z \mid x)q(z∣x).

Decoder: in deep learning, the decoder is a neural net that learns to reconstruct the data xx given a representation zz. In terms of probability models, the likelihood of the data xx given latent variables zz is parametrized by a generative network. The generative network outputs parameters to the likelihood distribution p(x \mid z)p(x∣z).

Inference: in neural nets, inference usually means prediction of latent representations given new, never-before-seen datapoints. In probability models, inference refers to inferring the values of latent variables given observed data.

VAE are built using the following six steps:

- An input image is passed through an encoder network.
- The encoder outputs parameters of a distribution Q(z|x).
- A latent vector z is sampled from Q(z|x). If the encoder learned to do its job well, most chances are z will contain the information describing x.
- The decoder decodes z into an image.
- Reconstruction error: the output should be similar to the input.
- Q(z|x) should be similar to the prior (multivariate standard Gaussian).

![](images/vae1.png)

![](images/vae2.png)

![](images/vae3.png)

![](images/vae4.png)

## Citing

```
@misc{Abhinav:2020,
  Author = {Abhinav Sagar},
  Title = {Bayesian Deep Learning},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/abhinavsagar/bayesian-deep-learning}}
}
```
