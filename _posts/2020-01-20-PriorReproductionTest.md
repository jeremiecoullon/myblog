---
layout: post
title: "Testing MCMC code: the prior reproduction test"
date: 2020-01-20 11:00:00 +0000
categories: MCMC statistics programming
---


[MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) is a class of algorithms for sampling from probability distributions. You can read about Metropolis-Hastings type algorithms [here](https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/). MCMC is a powerful algorithm, but is easy to get wrong and obtain samples from the wrong probability distribution. What's more, it won't be obvious if it fails, so we need ways to check whether the sampler is working correctly.

There are two main ways MCMC can fail: the chain doesn't mix and there is a bug in the software. We say that a chain mixes if given any starting point it eventually settles on the target distribution (namely, the posterior distribution we're looking for). To check that a chain mixes, we use diagnostics such as running the chain for a long time and examining the trace plots, calculating the R_hat, and using the multistart heuristics. See the [Handbook of MCMC](https://www.mcmchandbook.net/) for a good overview of these diagnostics.

To check whether there is a bug in the software, we can do tests such as unit tests which check that individual functions act like they should. A good integration test to do is to generate data given some fixed parameter (using yorur data model) and sample from the resulting posterior. The true parameter that generated the data should be within the posterior (loosely within 2 standard deviations of it). This checks that the sampler can indeed recover the true parameter!

However this test doesn't check whether the uncertainty is correct. Does the posterior have too much or too little variance due to a bug ? Or is the shape of posterior completely wrong ? A powerful way to test these statistical questions is the prior reproduction test. I have had trouble finding books or articles written about this (there are few, but none I've use this name); if you know of any references let me know! I know of this from my supervisor [Yvo Pokern](https://www.ucl.ac.uk/statistics/people/yvopokern) who learn it from [Gareth Roberts](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/roberts/); it seems that this method has often been transmitted by word of mouth.

The prior reproduction test runs as follows: sample from the prior \(\theta_0 \sim \pi\), generate data using this prior sample \(X \sim p(X|\theta_0)\), and run the to-be-tested sampler long enough to get an independent sample from the posterior \(\theta_p \sim \pi(\theta|X)\). The sample from the posterior should be distributed according the prior (derivation below); one can repeat this procedure to obtain many samples \(\theta_p\) and test whether they are distributed according to the prior.


Here is the test in code. First we define the data model (also sometimes called the observation operator \(\mathcal{G}\)) along with the log_likelihood, log_prior, and log_posterior.

```python
def G(theta):
	"G(theta): observation operator. Here it's just the identity function"
	return theta


# data noise:
sigma_data = 3
def build_log_likelihood(data_array):
	"Builds the log_likelihood function given some data"
	def log_likelihood(theta):
		"Data model: y = G(theta) + eps"
		return - (0.5)/(sigma_data**2) * np.sum([(elem - G(theta))**2 for elem in data_array])
	return log_likelihood

def log_prior(theta):
	# uniform prior on [0, 10]
	if not (0 < theta < 10):
		return -9999999
	else:
		return np.log(0.1)

def build_log_posterior(log_likelihood):
	def log_posterior(theta):
		return log_prior(theta) + log_likelihood(theta)
	return log_posterior
```

We want to the test the code for a sampler (given [here](location) in the `MCMC` module), so we run the PRT for it.

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

We then check that the posterior samples are uniformly distributed (ie: the prior):

<figure style="text-align:center">
  <img src="/assets/empirical_CDF_data10.png" style="width:96%; margin-left:2%; height:400px;"></iframe>
  <figcaption>Empirical CDF of the output of PRT: this seems to be uniformly distributed</figcaption>
</figure>


Note how we let the sampler run for 200 iterations to make sure that the posterior sample we get is independent of the initial condition. The number of iterations used needs to be tuned to the sampler; if it's slow then you'll need more samples! this means that the PRT will become more computationall expensive. We also needed to tune the proposal variance in the Gaussian proposal (called `sd_proposal`); ideally this will be a good tuning for any dataset generated in the PRT, but this may not always be the case! Sometimes the sampler needs hand tuning for each generated dataset; in this case it may also be too expensive to run the entire test. We'll see later what other tests we can do in this case.

Finally, how do we choose the number of data to generate (here we chose `10` datapoints) ? Consider 2 extremes: if we choose too much data then the posterior will have a very low variance and will be centered around the true parameter. So the posterior sample we obtain will be pretty much the true parameter we sampled from the prior, and the PRT will (trivially) produce samples from the prior. This however doesn't test the statistical properties of the sampler, but rather than the posterior mean is the true parameter. In the other extreme case, if we have too little data the likelihood will have a weak effect on the posterior which will essentially be the prior. The MCMC sampler will then sample from what is very close to prior, and again we are not testing our code. We need to choose somewhere in the middle so that the PRT tests the higher order moments of the posterior.

To tune the amount of data to generate we can plot the posterior vs the prior samples from the PRT as we can see in the figure below. Ideally there is a nice amount of variation around the line `y=x` as in the middle plot (for `N=10` data points). In the other two case the PRT will trivially recover prior samples and not test the software.

<figure style="text-align:center">
  <img src="/assets/3_data_comparison.png" style="width:96%; margin-left:2%; height:400px;"></iframe>
  <figcaption>Tune the amount of data to generate in PRT</figcaption>
</figure>

The proof of the PRT is as follows:



## Limitations

- proof
- intuition. If generate more/less data
- problem: expensive. Might need to hand tune the sample for each generated dataset
- other limitation: if there's a mode that the sampler mises, the PRT might not detect this.
