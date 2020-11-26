---
layout: post
title: "Ensemble samplers can sometimes work in high dimensions"
date: 2020-11-10 08:00:00 +0000
categories: MCMC statistics
---

A few years ago [Bob Carpenter](https://bob-carpenter.github.io/) wrote a fantastic [post](https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/) on how why ensemble methods cannot work in high dimensions. He explains how in high dimensions proposals that interpolate and extrapolate among samples are unlikely to fall in the typical set, causing the sampler to fail. During my PhD I worked on sampling problems with infinite-dimension parameters (ie: functions) and after reading Bob's post I always wondered if there was a way around this problem.

I finally got round to working on this idea and wrote a [paper](https://arxiv.org/pdf/2010.15181.pdf) (in review) with [Robert J Webber](https://cims.nyu.edu/~rw2515/) about an ensemble method for function spaces (ie: infinite dimensional parameters) which works very well. So does that mean that Bob was wrong about ensemble samplers? Of course not: it actually turns out that not all high dimensional distributions are the same. Namely: there is sometimes a low-dimensional subset of parameter space that represents all the "interesting bits" of the posterior that you can focus on.

In this post I go over how the functional ensemble sampler (FES) works and then discuss what this means for ensemble samplers in high dimensions.


# Functional ensemble sampler

## A need for these samplers

In many applied problems such as climate science or medical imaging, we are interested in approximating distributions on infinite-dimensional spaces. An example would be the problem in recovering the initial condition of a PDE given some noisy observations: this initial condition might be a function or a field.

EXPLAIN THE PROBLEM:

- Path reconstruction in a single well potential
- RW and the curse of dimensionality. This is related to Bob's point: in high dimensions random walk points away from the typical set.
- pCN: do a weighted average of previous sample and prior. For 1D: can check that the proposal has the same distribution as the prior.
- For function spaces, the prior is essential, as there is finite data but infinite dimensions. This is like GP regression.


There is a whole body of research for estimating these distributions using MCMC or SMC, but these methods usually need gradients. For lots of problems involving functions/fields, you don't have gradient: black box models written in Fortran. Impractical to rewrite in Stan/Pymc3/etc..

- Introduce AIES: extrapolate and interpolate. See Bob's post for why they fail in high dimensions

## FES

The idea of the functional ensemble sampler is do an eigen-expansion of the prior and use that to isolate a low-dimensional subspace that includes the "difficult bit" of the posterior. This subspace might have very correlated components or be nonlinear. The rest of the space (ie: the complementary subspace) is assumed to look like the prior. We can therefore alternate using AIES to sample from this space, and use pCN to sample from the complementary subspace. This is detailed as "algorithm 1" in the [paper](https://arxiv.org/pdf/2010.15181.pdf).

Note that this method only uses the prior to isolate this low-dimensional subspace. If you have gradients there is a principled way to use information from the model to find this: this is called the likelihood informed subspace (see [this paper](https://arxiv.org/abs/1403.4680))


Algorithm:

- step 1
- step 2

## Performance

We try out the algorithm to estimate the posterior path $$X_t$$ along with parameters $$\alpha$$ and $$\sigma$$ from the oscillator Langevin dynamics:

$$
\begin{cases}
dX = P dt \hspace{25mm} t >0\\
dP = -\alpha X dt + dW \hspace{5mm} t >0\\
X = P = 0 \hspace{24mm} t=0\\
\end{cases}
$$

We set


# Discussion
