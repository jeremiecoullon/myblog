---
layout: post
title: "Testing MCMC code: the prior reproduction test"
date: 2030-01-20 11:00:00 +0000
categories: MCMC statistics programming
---



[MCMC](https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/) is a class of algorithms for sampling from probability distributions. These are powerful algorithms, but it's easy to go wrong and obtain samples from the wrong probability distribution. What's more, it won't be obvious if the sampler fails, so we need ways to check whether it's is working correctly.

## MCMC

There are two main ways MCMC can fail: the chain doesn't mix and there is a bug in the software. We say that a chain mixes if given any starting point it eventually settles on the target distribution (namely, the posterior distribution we're looking for). To check that a chain mixes, we use diagnostics such as running the chain for a long time and examining the trace plots, calculating the R_hat, and using the multistart heuristics. See the [Handbook of MCMC](https://www.mcmchandbook.net/) for a good overview of these diagnostics.

To check whether there is a bug in the software, we can do tests such as unit tests which check that individual functions act like they should. A good integration test to do is to generate data given some fixed parameter (using your data model) and sample from the resulting posterior. The true parameter that generated the data should be within the posterior (loosely within 2 standard deviations of it). This checks that the sampler can indeed recover the true parameter!

However this test doesn't check whether the uncertainty is correct. Does the posterior have too much or too little variance due to a bug ? Or is the shape of posterior completely wrong ? A powerful way to test these statistical questions is the prior reproduction test. I have had trouble finding books or articles written about this (there are few, but none I've use this name); if you know of any references let me know! I know of this test from my PhD supervisor [Yvo Pokern](https://www.ucl.ac.uk/statistics/people/yvopokern) who learn it from [Gareth Roberts](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/roberts/). From talking to other researchers, it seems that this method has often been transmitted by word of mouth rather than from textbooks.


## The Prior Reproduction Test

The prior reproduction test runs as follows: sample from the prior $$ \theta_0 \sim \pi $$, generate data using this prior sample $$ X \sim p(X|\theta_0) $$, and run the to-be-tested sampler long enough to get an independent sample from the posterior $$ \theta_p \sim \pi(\theta|X) $$. The sample from the posterior should be distributed according the prior (derivation below);
one can repeat this procedure to obtain many samples $$ \theta_p $$ and test whether they are distributed according to the prior.

Here is the test in Python (code available on [Github](https://github.com/jeremiecoullon/PRT_post)). First we define the observation operator $$ \mathcal{G} $$) (the mapping from parameter to data, in the case simply the idenity) along with the log_likelihood, log_prior, and log_posterior. So here our data is simply sampled from a Gaussian with mean 5 and standard deviation 3.
```python
def G(theta):
	"""
	G(theta): observation operator. Here it's just the identity function, but it could
	be a more complicated model.
	"""
	return theta

# data noise:
sigma_data = 3

def build_log_likelihood(data_array):
	"Builds the log_likelihood function given some data"
	def log_likelihood(theta):
		"Data model: y = G(theta) + eps"
		return - (0.5)/(sigma_data**2)
			* np.sum([(elem - G(theta))**2 for elem in data_array])
	return log_likelihood

def log_prior(theta):
	"uniform prior on [0, 10]"
	if not (0 < theta < 10):
		return -9999999
	else:
		return np.log(0.1)

def build_log_posterior(log_likelihood):
	"Builds the log_posterior function given a log_likelihood"
	def log_posterior(theta):
		return log_prior(theta) + log_likelihood(theta)
	return log_posterior
```

We want to the test the code for a Metropolis sampler with Gaussian proposal (given in the [`MCMC` module](https://github.com/jeremiecoullon/PRT_post/tree/master/MCMC)), so we run the PRT for it (the following code is in the `run_PRT()` function in [`PRT.py`](https://github.com/jeremiecoullon/PRT_post/blob/master/PRT.py)):

```python
results = []
B = 200

for elem in range(B):
	# sample from prior
	sam_prior = np.random.uniform(0,10)

	# generate data points using the sampled prior
	data_array = G(sam_prior) + np.random.normal(loc=0, scale=sigma_data, size=10)

	# build the posterior function
	log_likelihood = build_log_likelihood(data_array=data_array)
	log_posterior = build_log_posterior(log_likelihood)

	# define the sampler
	ICs = {'theta': 1}
	sd_proposal = 20
	mcmc_sampler = MHSampler(log_post=log_posterior, ICs=ICs, verbose=0)
	# add a Gaussian proposal
	mcmc_sampler.move = GaussianMove(ICs, cov=sd_proposal)

	# Get a posterior sample.
	# Let the sampler run for 200 iterations to make sure it's independent from the initial condition
	mcmc_sampler.run(n_iter=200, print_rate=300)
	last_sample = mcmc_sampler.all_samples.iloc[-1].theta

	# store the results. Keep the posterior sample as well as the prior that generated the data
	results.append({'posterior': last_sample, 'prior': sam_prior})
```

We then check that the posterior samples are uniformly distributed (ie: same as the prior):

<figure class="post_figure">
  <img src="/assets/PRT_post/empirical_CDF_data10.png">
  <figcaption>Empirical CDF of the output of PRT: these seems to be uniformly distributed</figcaption>
</figure>


## Tuning the PRT

Notice how we let the sampler run for 200 iterations to make sure that the posterior sample we get is independent of the initial condition (`mcmc_sampler.run(n_iter=200, print_rate=300)`). The number of iterations used needs to be tuned to the sampler; if it's slow then you'll need more samples. This means that a slowly mixing sampler will cause the PRT to become more computationally expensive. We also needed to tune the proposal variance in the Gaussian proposal (called `sd_proposal`); ideally this will be a good tuning for any dataset generated in the PRT, but this may not always be the case. Sometimes the sampler needs hand tuning for each generated dataset; in this case it may also be too expensive to run the entire test. We'll see later what other tests we can do in this case.

Finally, how do we choose the amount of data to generate (here we chose `10` datapoints) ? Consider 2 extremes: if we choose too much data then the posterior will have a very low variance and will be centered around the true parameter. So the posterior sample we obtain will be pretty much the true parameter we sampled from the prior, and the PRT will (trivially) produce samples from the prior. This however doesn't test the statistical properties of the sampler, but rather than the posterior mean is the true parameter. In the other extreme case, if we have too little data the likelihood will have a weak effect on the posterior, which will then essentially be the prior. The MCMC sampler will then sample from a distribution that is very close to prior, and again we are not testing our code. We therefore need to choose somewhere in the middle.

To tune the amount of data to generate we can plot the posterior vs the prior samples from the PRT as we can see in the figure below. Ideally there is a nice amount of variation around the line `y=x` as in the middle plot (for `N=10` data points). In the other two case the PRT will trivially recover prior samples and not test the software properly.

<figure class="post_figure">
  <img src="/assets/PRT_post/3_data_comparison.png">
  <figcaption> We need to tune the amount of data to generate in PRT</figcaption>
</figure>




## Limitations

In some cases however it's not possible to run the PRT. The likelihood may be too computationally expensive; it might require solving numerically a differential equation for example. It's also possible that the proposal distribution needs to be tuned for each dataset.
In this case you have to tune the proposal manually at each iteration of the PRT.

A way to deal with these problems is to only test conditionals of the posterior (in the case of higher dimensional posteriors).
For example if the posterior is $$\pi(\theta_1, \theta_2)$$, then run the test on $$ \pi(\theta_1 | \theta_2) $$. In some cases this can solve the problem of needing to retune the proposal distribution for every dataset. This also helps with problem of expensive forward likelihood, as the dimension of the conditional posterior is lower than the original one. Less samples are then needed to run the test.


Another very simple alternative is to use the sampler to sample from the prior (so simply commenting out the likelihood function in the posterior). This completely bypasses the problem of expensive likelihoods and the need to retune the proposal at every step. This test checks that the MCMC proposal is correct (the Hastings correction for example), so is good for complicated proposals. However if the proposal needed to sample from the prior is qualitatively different from the proposal needed to sample from the posterior, then it's not a useful test.

_Code to reproduce the figures is on [Github](https://github.com/jeremiecoullon/PRT_post)_
