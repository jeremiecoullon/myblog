---
layout: post
title: "Ensemble samplers can sometimes work in high dimensions"
date: 2021-02-10 08:00:00 +0000
categories: MCMC statistics
---

A few years ago [Bob Carpenter](https://bob-carpenter.github.io/) wrote a fantastic [post](https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/) on how why ensemble methods cannot work in high dimensions. He explains how high dimensions proposals that interpolate and extrapolate among samples are unlikely to fall in the typical set, causing the sampler to fail. During my PhD I worked on sampling problems with infinite-dimension parameters (ie: functions) and intractible gradients. After reading Bob's post I always wondered if there was a way around this problem.

I finally got round to working on this idea and wrote a [paper](https://arxiv.org/pdf/2010.15181.pdf) (to appear in Statistics and Computing) with [Robert J Webber](https://cims.nyu.edu/~rw2515/) about an ensemble method for function spaces (ie: infinite dimensional parameters) which works very well. So does that mean that Bob was wrong about ensemble samplers? Of course not; it rather turns out that not all high dimensional distributions are the same. Namely: there is sometimes a low-dimensional subset of parameter space that represents all the "interesting bits" of the posterior that you can focus on.

In this post I'll start off by going over how MCMC samplers can be defined on function spaces before introducing the functional ensemble sampler (FES). I'll then discuss what this means for using gradient-free samplers such as ensemble samplers in high dimensional spaces. You can skip directly to the [discussion section](#discussion) for a reply to Bob's post, which can be read independently of the description of the FES algorithm.


# Functional space samplers

## The need for these samplers

In many applied problems such as climate science or medical imaging, we are interested in sampling from distributions on infinite-dimensional spaces. An example would be the problem in recovering the initial condition of a PDE given some noisy observations: this initial condition might be a function or a field.

Consider the 1D advection equation, a linear hyperbolic PDE. This PDE models how a quantity - such as the density of fluid - propagates through 1D domain over time. This PDE is a special case of more complicated nonlinear PDEs that arise in fluid dynamics, acoustic, motorway traffic flow, and many more applications. The equation is given as follows (with subscripts denoting partial differentiation):

$$\rho_t + c \rho_x = 0$$

Here $$\rho$$ is density and $$c \in \mathcal{R}$$ is the wave speed of the fluid. We define the initial condition $$\rho_0 \equiv \rho_0(x)$$ to be the state of density of the fluid at time $$t=0$$. This linear PDE simply advects the initial condition to the right or left of the domain depending on the sign of $$c$$.

So the solution can be written as $$\rho(x,t) = \rho(x-ct)$$. The following figure shows how an initial density profile is advected (ie: transported) to the right with wavespeed $$c=1$$.



<figure class="post_figure post_figure_larger">
  <img src="/assets/FES_post/advection_solution.png">
  <figcaption>Figure 1: The left-most panel shows the initial condition. The next 2 panels show it being transported to the right with speed c=1</figcaption>
</figure>

An inverse problem might be: given some noisy observations of _flow_ at a few locations in space (these could correspond to detectors along a pipe for example), recover the initial condition as well as the wave speed of the fluid. Note that flow is the product of density and speed: $$q = \rho c$$: so we're observing flow and inferring the latent parameters $$\rho$$ (a function) and $$c$$ (a scalar).


To solve, we need to set a prior on both parameters, so we choose a uniform prior for the wave speed, and a Gaussian Process (GP) prior for the initial condition. Note that if the wave speed is known this becomes standard GP regression.



## A naive try


Let's see what we can do to sample from discretisations of functions such as $$\rho_0$$. Consider the problem of sampling from the posterior $$\pi(x) \sim L(x) \pi_0(x)$$ with $$L(x)$$ the data likelihood, $$\pi_0(x)\sim \mathcal{N}(0, \Sigma_0)$$ the prior, and $$x \in \mathcal{R}^N$$ is a discretisation of $$\rho_0$$.

If we try a standard gaussian proposal: $$\tilde{x} = x + \xi$$ with $$\xi \sim \mathcal{N}(0,\Sigma_0)$$, then the curse of dimensionality hits and the acceptance rate goes to zero as $$N \to \infty$$. This happens for the reason explained by Bob in his [post](https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/). Namely, the posterior is a very thin shell in a high dimensional space and random walk makes a random proposal that will always be pointing away from the typical set of the distribution (in particular it will be pointing away from the mean).

Could we then use sophisticated samplers such as HMC to sample from this posterior? We could indeed do this for moderately sized $$N$$, but as we increase the resolution of our discretisation the acceptance rate of these samplers will also eventually go to zero.

This is not ideal, as we would like samplers that keep working in the limit of infinite resolution. This is similar to the fact that (good) numerical methods for ODEs or PDEs tend to the correct solution as we increase the resolution (ie: the resolution in time or space).


## pCN


Let us now consider the function space version of random walk: preconditioned Crank Nicholson (see [this paper](https://arxiv.org/abs/1202.0709) for a nice overview of function space samplers). Consider the following proposal, with $$u$$ current MCMC sample, $$\beta \in (0,1)$$ and $$\xi \sim \mathcal{N}(0, \Sigma_0)$$ a sample _from the prior_:

$$\tilde{x} = \sqrt{1-\beta^2}x + \beta \xi$$


This proposal is asymmetric, so we need a Hastings correction. It turns out that this correction cancels out exactly with the prior. The acceptance probability is then (with $$L(x)$$ the likelihood):

$$\alpha = \text{min}\{1, \frac{L(\tilde{x})}{L(x)} \}$$

Notice that the dimensionality of the parameter doesn't appear in the acceptance probability, and so this sampler is now freed from the curse of dimensionality. The sampler may still of course mix slowly, but that will only be because of the likelihood.

We note that this independence of dimension is only possible because we have a prior (a GP prior) that imposes a lot of structure on the parameter. Namely: points close together are very correlated.

There is a whole literature on algorithms that improve on pCN to sample from these distributions, but these methods usually need gradients (ex: functional MALA or functional HMC). However for lots of problems involving functions or fields, you don't have gradient: you might have large codebases written in Fortran that are impractical to rewrite in Stan/Pymc3/etc. This is why Robert Webber and I developed a sampler for function space that doesn't use gradients, called the [functional ensemble sampler](https://arxiv.org/pdf/2010.15181.pdf) (FES).

## Functional ensemble sampler

### the sampler

The idea of the functional ensemble sampler is do an eigenfunction expansion of the prior and use that to isolate a low-dimensional subspace that includes the "difficult bit" of the posterior. This means that we can represent functions $$u$$ in this space by truncating the eigenexpansion to $$M$$ basis elements: $$u_+ = \sum^{M} u_j \phi_j$$. This subspace might have very correlated components, be nonlinear, and more generally be difficult to sample from.

Functions in the rest of the space (ie: the complementary subspace) can be represented by $$u_- = \sum_{M+1}^{\infty} u_j \phi_j$$ and are assumed to look like the prior. We can therefore alternate using a finite dimensional sampler (we'll use the [affine invariant ensemble sampler](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)) to sample from this space, and use pCN to sample from the complementary subspace.

We now give a brief overview of the affine invariant ensemble sampler (AIES). We define an ensemble of walkers that each sample the target distribution. To update walker $$X_k$$, choose another walker $$X_j$$ in the ensemble and propose $$Y =  X_j(1-Z) + Z X_k$$, with $$Z$$ a univariate random variable with a suitable density. Given the correct accept/reject step the ensemble converges to the product space of the posterior.

One of the benefits of this sampler is that it's affine invariant and self tuning, ie: it adapts to the structure of the posterior. It also doesn't need any gradient information and can be easily parallelised so is well suited for problems that cannot be rewritten in frameworks like Stan or PyMC3. However it has the limitation that it doesn't work for high dimensions (ie: higher than 20, though this depends on the problem), as discussed by Bob in his [post](https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/).

So the functional ensemble sample run a Metropolis-within-Gibbs algorithm that alternates sampling from the low dimensional space using AIES and the complementary space using pCN. You can find more detail of the algorithm in the [paper](https://arxiv.org/pdf/2010.15181.pdf) (see algorithm 1).

### the likelihood informed subspace

Note that our dimensionality reduction method only uses the prior to isolate this low-dimensional subspace. If you have gradients there is a principled way to use information from the model to find it; this is called the [likelihood informed subspace](https://arxiv.org/abs/1403.4680) (LIS). The LIS is a linear subspace where the prior and posterior differ the most. This _finite dimensional_ subspace therefore corresponds to the bit of the posterior that's the most difficult to sample from, so this is where we should apply sophisticated samplers.

When we do our truncated eigenfunction expansion of the prior (as discussed above), we're trying to approximate this LIS. Indeed if we choose $$M$$ large enough we'll eventually be able to englobe the LIS. However this might require a value of $$M$$ that is too large which means that our AIES algorithm will struggle with the high dimensionality. As a result our truncated eigenfunction expansion might not be the $$M$$-dimensional subspace that best represents the "difficult" aspects of the posterior, but it might be good enough for the problem at hand.


## Performance

We go back to the advection equation and try out the algorithm.
Given the equation $$\rho_t + c\rho_x = 0$$, the inverse problem consists of inferring the wave speed $$c$$ and initial conditions $$\rho_0$$ from 9 noisy observations of _flow_ $$q$$. These observations come from equally spaced detectors at several time points. We emphasise that we observe flow data (not density), and that flow given by the relation $$q = \rho c$$.


We run FES with $$L=100$$ walkers and try different values of the truncation parameter: $$M \in \{0,1,5,10,20\}$$. We note that setting $$M=0$$ corresponds to using AIES for the wave speed $$c$$ and pCN for the initial condition. See the [paper](https://arxiv.org/pdf/2010.15181.pdf) for more details of this experiment.

We compare FES to a standard pCN sampler which uses the following joint update:
- Gaussian proposals for the wave speed $$c$$
- pCN for the initial condition $$\rho_0(x)$$

We run the samplers and show the ACF plots for the wave speed and the coefficient of the first eigenfunction in figure 2 below. The coefficient of the first eigenfunction corresponds to the first parameter in the new low-dimensional basis.

<figure class="post_figure">
  <img src="/assets/FES_post/advection_ACF.png">
  <figcaption>Figure 2: ACF curves for wave speed and the coefficient of the first eigenfunction</figcaption>
</figure>


We can see that the ensemble sampler can be up to two orders of magnitude faster than pCN. We can also see that there's an optimal value of $$M$$ to choose to get the fastest possible mixing.

Our [paper](https://arxiv.org/pdf/2010.15181.pdf) has more experiments and discussions about the performance and limitations of the sampler.

# Discussion

Bob Carpenter's explanation of why ensemble methods fail in high dimensions is of course correct: ensemble methods will fail in high dimensions because interpolating and extrapolating between points will fall outside the typical set. However my main point is that high dimensional spaces are not all high dimensional in the same way.

Throughout the post Bob uses a high dimensional Gaussian (ie: a doughnut) to illustrate when ensemble methods fail. Indeed, we statistician often use the Gaussian distribution to illustrate ideas or simplify reasoning. For example, a standard line of thinking when developing a new method might be to argue: _"in the case of everything being Gaussian our method becomes exact.."_. This makes sense because Gaussians are easy to manipulate and pop everywhere because of the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem).

However, it can be unhelpful to build our mental models of difficult concepts - such as high dimensional distributions - solely on Gaussians. Indeed, difficult sampling problems are hard precisely because they are not Gaussian. This is similar to the problem of using [model organsisms](https://en.wikipedia.org/wiki/Model_organism) in biology to learn about humans. So in a way, Gaussians are the [fruit flies](https://en.wikipedia.org/wiki/Drosophila_melanogaster) of statistics.

### high dimensional distributions

Here are two ways (among many!) in which high dimensional distributions might be different from a spherical Gaussian which we can use to help with sampling.

Firstly, there might exist a low dimensional subspace that represents most of the "interesting bits" of the posterior. The rest of the space (the complementary subspace) is then relatively unaffected by the likelihood and therefore acts like the prior. This is similar to the idea behind PCA where the data mainly lives on a low dimensional subspace.

In our inverse problem (see the description of the [FES algorithm](#the-sampler)) involving functional parameters, the prior gives a natural way to find this low dimensional subspace. This is because the Gaussian process prior imposes a lot of structure on the parameter which allows the infinite dimensional problem to be well posed. The LIS gives an even better approximation of this subspace, but requires gradients of the posterior (funnily enough, the construction of the LIS assumes a Gaussian posterior).
- Example in molecular dynamics? Ask Rob


Secondly, it might be possible to group the parameters together such that parameters within the group are correlated and parameters between the groups are fairly independent. This is similar to the first point above, but here the different groups might all be challenging to sample from. This is called Gibbs blocking, and Ter Braak did exactly this in his [paper on DE-MCMC](https://link.springer.com/article/10.1007/s11222-006-8769-1) to sample from high dimensional spaces (see section 5.1 of the paper: _"Crossover and block updating"_). Gibbs blocking will work if the blocks are fairly independent from each other, so the challenge is to find combinations of parameters that decompose the parameter space in this way. In Ter Braak's example (a nonlinear mixed-effects model) the blocks come naturally as the model has certain groupings of parameters that are expected to be independent.

If you have a moderately high dimensional problem (ie: dimension around greater than 20) and gradients are unavailable, you might need to explore your posterior to see if there are any important low dimensional subspaces that can be used. This might involve thinking about the physical interpretation of the parameters. Ie: do we expect certain parameters to be related to each other and others to be independent? Do we expect combinations of parameters to be constant?

# Conclusion

Making MCMC work is about finding a good parametrisation. HMC uses Hamilton's equations to find a natural parametrisation (ie: level sets). However, this can still break down with difficult geometries and another [reparametrisation](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) might be necessary.

If you don't have gradients then ensemble methods can be a good solution (example: the [emcee](https://emcee.readthedocs.io/en/stable/) package). However if the dimension is too high then these will break down as discussed in Bob Carpenter's [post](https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/). Thankfully, this is not the end of the road for ensemble samplers! You'll then need to think about your problem to identify natural groupings of parameters to apply your gradient-free sampler to. In this post I went over a practical way to do this in the case of infinite-dimensional inverse problems, yielding a simple but powerful gradient-free ensemble sampler defined on function spaces.
