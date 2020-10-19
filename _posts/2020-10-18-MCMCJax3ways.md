---
layout: post
title: "MCMC in Jax with benchmarks: 3 ways to write a sampler"
date: 2020-10-18 08:00:00 +0000
categories: MCMC jax statistics programming
---

<!-- todo: fix the "I" vs "we" throughout -->

In this post I show 3 ways to write a sampler using Jax. I found that although there are a bunch of tutorials about learning the basics of Jax, it was not clear to me what the best way was to write a sampler in Jax and what the tradeoff were in terms of speed. So this blog post goes over 3 ways to write a sampler and focuses on the speed of each sampler.

I'll assume that you already know some Jax, in particular the function `grad`, `vmap`, and `jit`. If not, you can check out how to use these in this [blog post](https://colinraffel.com/blog/you-don-t-know-jax.html) or in the [Jax documentation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)! I'll use features like the PRNG but will only briefly discuss how they work. I will rather focus on the different ways of using Jax for sampling and the speed performance of each version of the sampler.


## Sampler and model

To benchmark the samplers I use Bayesian logistic regression throughout. As sampler I use the unadjusted Langevin algorithm (ULA) with Euler dicretisation, as it is one of the simplest gradient-based samplers out there due to the lack of accept-reject step. Let $$\theta_n \in \mathcal{R}^d$$ be the parameter at iteration $$n$$,  $$\nabla \log \pi(\theta)$$ the gradient of the log-posterior, $$dt$$ the step size, and $$\xi \sim \mathcal{N}(0, I_d)$$. Given a current position of the chain, the next sample is given by the equation:

$$\theta_{n+1} = \theta_n + dt\nabla\log\pi(\theta_n) + \sqrt{2dt}\xi$$


The setup of the logistic regression model is the same as the one from this [SG-MCMC review paper](https://arxiv.org/abs/1907.06986):

- Matrix of covariates $$\textbf{X} \in \mathcal{R}^{N\times d}$$, and vector responses: $$\textbf{y} = \{ y_i \}_1^N$$
- Parameters: $$\theta \in \mathcal{R^d}$$


**Model:**

- $$y_i = \text{Bernoulli}(p_i)$$ with $$p_i = \frac{1}{ 1+\exp(-\theta^T x_i)}$$
- Prior: $$\theta \sim \mathcal{N}(0, \Sigma_{\theta})$$ with $$\Sigma_{\theta} = 10\textbf{I}_d$$
- Likelihood: $$ p(X,y \mid \theta) = \Pi^N p_i^{y_i}(1-p_i)^{y_i} $$



## Version 1: Jax for the log-posterior

In this version I only use Jax to write the log-posterior function (or the loss function in the case of optimisation). I use `vmap` to calculate the log-likelihood for each data point, `jit` to compile the function, and `grad` to get the gradient. The rest of the sampler is a simple Python loop with Numpy to store the samples, as is shown below:

```python
def ula_sampler_python(grad_log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = onp.zeros((num_samples, dim))
    paramCurrent = x_0
    current_grad = grad_log_post(paramCurrent)

    print(f"Python sampler:")
    for i in range(num_samples):
        paramGradCurrent = grad_log_post(paramCurrent)
        paramCurrent = paramCurrent + dt*paramGradCurrent +
                        onp.sqrt(2*dt)*onp.random.normal(size=(paramCurrent.shape))
        samples[i] = paramCurrent
        if i%print_rate==0:
            print(f"Iteration {i}/{num_samples}")
    return samples
```

In this sampler I write the udpate equation using Numpy and store the samples in the array `samples`. Also note that Numpy is imported as `onp` in this case ("ordinary Numpy").

## Version 2: Jax for the transition kernel

With Jax we can compile functions using `jit` which means they'll run faster (we did this for the log-posterior function). Could we not put the bit inside the loop in a function and compile that? The issue is that for `jit` to work, you can't have numpy arrays or use the numpy random number generator (`onp.random.normal()`).


Jax does random numbers a bit differently to Numpy. I won't explain how this bit works; you can read about them in the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG). The main idea is that you have to explicitly pass in a PRNG (called `key`) to every function that includes randomness, and split the key to get different pseudorandom numbers.

Below is a function for the transition kernel of the sampler rewritten to only include Jax functions and arrays (so it can be compiled). The point of the `partial` decorator and the `static_argnums` argument is to point to which arguments will not change once the function is compiled. Indeed, the function for the gradient of the log-posterior or the step size will not change throughout the sampler, but the PRNG key and the parameter definitely will! The means that the function will run faster as it can hardcode these static values/functions during compilation. Note that if the argument is a function (as is the case for `grad_log_post`) you must set it as static. See the [documentation](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) for info on this.

```python
@partial(jit, static_argnums=(2,3))
def ula_kernel(key, param, grad_log_post, dt):
    paramGrad = grad_log_post(param)
    param = param + dt*paramGrad + np.sqrt(2*dt)*random.normal(key=key, shape=(param.shape))
    return param
```

The main loop in the previous function now becomes:

```python
print(f"Python loop with Jax kernel")
for i in range(num_samples):
    key, subkey = random.split(key)
    param = ula_kernel(subkey, param, grad_log_post, dt)
    samples[i] = param
    if i%print_rate==0:
        print(f"Iteration {i}/{num_samples}")
```
Notice how we split the random key just before the `ula_kernel()` function, but still save the samples in the numpy array `samples`. Running this function several times with the same starting PRNG key will now produce exactly the sample samples, which means that the sampling is completely reproducible.

## Version 3: full Jax  

We've written more of our function in Jax, but there is still some Python left. Could we rewrite the entire sampler in Jax? It turns out that we can! Jax allows us to do [for loops](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html), but because Jax are designed to work on [pure functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Pure-functions) you need to use for [`scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) function. Note that Jax also has a similar [`fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) function which apparnetly you should only use if you can't use scan (see [discussion](https://github.com/google/jax/discussions/3850). In the case of our sampler `scan` is easier to use as you don't need to keep track of the entire chain of samples; `scan` does it for you (in `fori_loop` you have to pass an array of samples in `state` which you update as you go along). In terms of performance I did quick benchmark for both and didn't see a speed difference in this case, though the [discussion](https://github.com/google/jax/discussions/3850) says there can be speed benefits.

The way to use `scan` is to pass in a function that is called at every iteration. This function takes in `carry` which contains all the information you use in each iteration (and which you update as you go along). It also takes in `x` which is the value of the array you're iterating over. It should return an updated version of `carry` along with anything who's progress you want to keep track of (in our case, we want to store all the samples as we iterate.

Here is the function that we'll pass in `scan`. Note that the first line unpacks `carry`. The key is then split and the new parameter is generated using the `ula_kernel` function used above.

```python
def ula_step(carry, x):
  key, param = carry
  key, subkey = random.split(key)
  param = ula_kernel(subkey, param, grad_log_post, dt)
  return (key, param), param
```

You can then pass this function along with the inital state in `scan`, and recover the final `carry` along with all the samples. The last two arguments in `scan` mean that we don't care what we're iterating over; we simply want to run the sampler for `num_samples` number of iterations (as always, see the [docs](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) for details).

```python
carry = (key, x_0)
carry, samples = lax.scan(ula_step, carry, None, num_samples)
```

Putting it all together in a single function, we get the following. Notice that we compile the entire function (with `grad_log_post`, `num_samples`, `dt` kept as static).

```python
@partial(jit, static_argnums=(1,2,3))
def ula_sampler_full_jax_jit(key, grad_log_post, num_samples, dt, x_0):

    def ula_step(carry, x):
        key, param = carry
        key, subkey = random.split(key)
        param = ula_kernel(subkey, param, grad_log_post, dt)
        return (key, param), param

    carry = (key, x_0)
    _, samples = lax.scan(ula_step, carry, None, num_samples)
    return samples
```

Having the entire function written in Jax means that once the function is compiled it will usually be faster (see benchmarks below), and we can rerun it for different PRNG keys to get different realisations of the chain. We can also run this function in `vmap` (mapping over the keys or inital conditions) to get several chains running in parallel. Check out this [blog post](https://rlouf.github.io/post/jax-random-walk-metropolis/) for a benchmark of a Metropolis sampler in parallel using Jax and Tensorflow.

## Benchmarks
