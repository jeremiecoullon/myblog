---
layout: post
title: "MCMC in Jax with benchmarks: 3 ways to write a sampler"
date: 2020-10-30 08:00:00 +0000
categories: MCMC jax statistics programming
---


This post goes over 3 ways to write a sampler using Jax. I found that although there are a bunch of tutorials about learning the basics of Jax, it was not clear to me what was the best way to write a sampler in Jax. In particular, how much of the sampler should you write in Jax? Just the log-posterior (or the loss in the case of optimisation), or the entire loop? This blog post tries to answer this by going over 3 ways to write a sampler while focusing on the speed of each sampler.

<!-- todo: fix the "I" vs "we" throughout -->

<!-- Add github link in 2 places -->

I'll assume that you already know some Jax, in particular the functions `grad`, `vmap`, and `jit`, along with the random number generator. If not, you can check out how to use these in this [blog post](https://colinraffel.com/blog/you-don-t-know-jax.html) or in the [Jax documentation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)! I will rather focus on the different ways of using Jax for sampling (using the ULA sampler) and the speed performance of each implementation. I'll then redo these benchmarks for 2 other samplers (MALA and SLGD) You can find the code to reproduce all these examples on [Github](link!).


## Sampler and model

To benchmark the samplers we'll Bayesian logistic regression throughout. As sampler we'll start with the unadjusted Langevin algorithm (ULA) with Euler dicretisation, as it is one of the simplest gradient-based samplers out there due to the lack of accept-reject step. Let $$\theta_n \in \mathcal{R}^d$$ be the parameter at iteration $$n$$,  $$\nabla \log \pi(\theta)$$ the gradient of the log-posterior, $$dt$$ the step size, and $$\xi \sim \mathcal{N}(0, I_d)$$. Given a current position of the chain, the next sample is given by the equation:

$$\theta_{n+1} = \theta_n + dt\nabla\log\pi(\theta_n) + \sqrt{2dt}\xi$$


The setup of the logistic regression model is the same as the one from this [SG-MCMC review paper](https://arxiv.org/abs/1907.06986):

- Matrix of covariates $$\textbf{X} \in \mathcal{R}^{N\times d}$$, and vector responses: $$\textbf{y} = \{ y_i \}_1^N$$
- Parameters: $$\theta \in \mathcal{R^d}$$


**Model:**

- $$y_i = \text{Bernoulli}(p_i)$$ with $$p_i = \frac{1}{ 1+\exp(-\theta^T x_i)}$$
- Prior: $$\theta \sim \mathcal{N}(0, \Sigma_{\theta})$$ with $$\Sigma_{\theta} = 10\textbf{I}_d$$
- Likelihood: $$ p(X,y \mid \theta) = \Pi^N p_i^{y_i}(1-p_i)^{y_i} $$



## Version 1: Python loop with Jax for the log-posterior

In this version we only use Jax to write the log-posterior function (or the loss function in the case of optimisation). We use `vmap` to calculate the log-likelihood for each data point, `jit` to compile the function, and `grad` to get the gradient (see the code on [Github](link!)). The rest of the sampler is a simple Python loop with NumPy to store the samples, as is shown below:

```python
def ula_sampler_python(grad_log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = onp.zeros((num_samples, dim))
    paramCurrent = x_0

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

In this sampler we write the udpate equation using NumPy and store the samples in the array `samples`. Note that NumPy is imported as `onp` ("ordinary NumPy").

## Version 2: Jax for the transition kernel

With Jax we can compile functions using `jit` which makes them run faster (we did this for the log-posterior function). Could we not put the bit inside the loop in a function and compile that? The issue is that for `jit` to work, you can't have NumPy arrays or use the NumPy random number generator (`onp.random.normal()`).


Jax does random numbers a bit differently to NumPy. I won't explain how this bit works; you can read about them in the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG). The main idea is that you have to explicitly pass in a PRNG (called `key`) to every function that includes randomness, and split the key to get different pseudorandom numbers.

Below is a function for the transition kernel of the sampler rewritten to only include Jax functions and arrays (so it can be compiled). The point of the `partial` decorator and the `static_argnums` argument is to point to which arguments will not change once the function is compiled. Indeed, the function for the gradient of the log-posterior or the step size will not change throughout the sampler, but the PRNG key and the parameter definitely will! The means that the function will run faster as it can hardcode these static values/functions during compilation. Note that if the argument is a function (as is the case for `grad_log_post`) you don't have a choice and must set it as static. See the [documentation](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) for info on this.

```python
@partial(jit, static_argnums=(2,3))
def ula_kernel(key, param, grad_log_post, dt):
    key, subkey = random.split(key)
    paramGrad = grad_log_post(param)
    param = param + dt*paramGrad + np.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    return key, param
```

The main loop in the previous function now becomes:

```python
for i in range(num_samples):
    key, param = ula_kernel(key, param, grad_log_post, dt)
    samples[i] = param
    if i%print_rate==0:
        print(f"Iteration {i}/{num_samples}")
```
Notice how we split the random key inside `ula_kernel()` function which means it gets compiled (Jax's random number generator can be [slow in some cases](https://github.com/google/jax/issues/968)). We still save the samples in the NumPy array `samples` as in the previous case. Running this function several times with the same starting PRNG key will now produce exactly the sample samples, which means that the sampler is completely reproducible.

## Version 3: full Jax  

We've written more of our function in Jax, but there is still some Python left. Could we rewrite the entire sampler in Jax? It turns out that we can! Jax does allow us write loops, but as it is designed to work on [pure functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Pure-functions) you need to use the [`scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) function. This function which allows you to loop over an array (similar to doing `for elem in mylist` in Python).

The way to use `scan` is to pass in a function that is called at every iteration. This function takes in `carry` which contains all the information you use in each iteration (and which you update as you go along). It also takes in `x` which is the value of the array you're iterating over. It should return an updated version of `carry` along with anything who's progress you want to keep track of: in our case, we want to store all the samples as we iterate.

Note that Jax also has a similar [`fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) function which apparently you should only use if you can't use scan (see the [discussion on Github](https://github.com/google/jax/discussions/3850)). In the case of our sampler `scan` is easier to use as you don't need to explicitly keep track of the entire chain of samples; `scan` does it for you. In contrast, when using `fori_loop` you have to pass an array of samples in `state` which you update yourself as you go along. In terms of performance I did quick benchmark for both and didn't see a speed difference in this case, though the [discussion on Github](https://github.com/google/jax/discussions/3850) says there can be speed benefits.


Here is the function that we'll pass in `scan`. Note that the first line unpacks `carry`. The `ula_kernel` function then generates the new key and parameter. We then return the new version of `carry` (ie: `(key, param)`) which includes the updated key and parameter, and return the current parameter (`param`) which `scan` will save in an array.

```python
def ula_step(carry, x):
  key, param = carry
  key, param = ula_kernel(key, param, grad_log_post, dt)
  return (key, param), param
```

You can then pass this function along with the initial state in `scan`, and recover the final `carry` along with all the samples. The last two arguments in the `scan` function below mean that we don't care what we're iterating over; we simply want to run the sampler for `num_samples` number of iterations (as always, see the [docs](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) for details).

```python
carry = (key, x_0)
carry, samples = lax.scan(ula_step, carry, None, num_samples)
```

Putting it all together in a single function, we get the following. Notice that we compile the entire function with `grad_log_post`, `num_samples`, and `dt` kept as static. We allow the PRNG key and the starting point of the chain `x_0` to vary so we can get different realisations of our chain.

```python
@partial(jit, static_argnums=(1,2,3))
def ula_sampler_full_jax_jit(key, grad_log_post, num_samples, dt, x_0):

    def ula_step(carry, x):
        key, param = carry
        key, param = ula_kernel(key, param, grad_log_post, dt)
        return (key, param), param

    carry = (key, x_0)
    _, samples = lax.scan(ula_step, carry, None, num_samples)
    return samples
```

Having the entire function written in Jax means that once the function is compiled it will usually be faster (see benchmarks below), and we can rerun it for different PRNG keys or different initial conditions to get different realisations of the chain. We can also run this function in `vmap` (mapping over the keys or inital conditions) to get several chains running in parallel. Check out this [blog post](https://rlouf.github.io/post/jax-random-walk-metropolis/) for a benchmark of a Metropolis sampler in parallel using Jax and Tensorflow.

However a disadvantage of doing this is that I don't you can't (AFAIK) print a progress bar (ex: print `"iteration 1000/10000"`). A simple fix is to have the Jax loop run a tenth of the samples (say) and loop those in a Python while concatenating the results. That way you can print the progress bar at each Python loop iteration, while keeping most of the speed benefits of Jax. You also would get the compilation benefits in the first run of the sampler, as the Jax sampler would get compiled in the first Python loop iteration.

# Benchmarks

Now that we've gone over 3 ways to write an MCMC sampler we'll show some speed benchmarks for ULA along with two other algorithms. We use the logistic regression model presented above and run `20 000` samples throughout.


## Unadjusted Langevin algorithm

### Increase amount of data:

We run ULA for `20 000` samples for a 5 dimensional parameter. We vary the amount of data used and see how fast the algorithms are (time is in seconds).

dataset size | python | Jax kernel | full Jax (1st run) | full Jax (2nd run)
--- | --- | --- | --- | ---
$$10^3$$ | 11 | 3.4 | 0.53 | 0.18
$$10^4$$ | 11 | 4.6 | 2.0 | 1.6
$$10^5$$ | 32 | 32 | 24 | 24
$$10^6$$ | 280 | 280 | 250 | 250


We can see that for small amounts of data the full Jax sampler is much faster than the Python loop. In particular, for 1000 data points the full Jax sampler (once compiled) is almost 60 times faster than the Python loop version.

Note that all the samplers use Jax to get the gradient of the log-posterior (including the Python loop version). So the speedup comes from everything else in the sampler being compiled. We also notice that for small amounts of data, there's a big difference between the first full Jax run (where the function is being compiled) and the second one (where the function is already compiled). This speedup would be especially useful if you need to run the sampler many times for different starting points or realisation (ie: choosing a different PRNG key). We can also see that simply writing the transition kernel in Jax already causes a 3x speedup over the Python loop version.

However as we add more data, the differences between the algorithms gets smaller. The full Jax version is still the fastest, but not by much. This is probably because the log-posterior dominates the computational cost of the sampler as the dataset increases. As that function is the same for all samplers, they end up having similar timings.


### Increase the dimension:

We now run the samplers with a fixed dataset size of 1000 data points, and run each sampler for 20K iterations while varying the dimension:


dimension | python | Jax kernel | full Jax (1st run) | full Jax (1nd run)
--- | --- | --- | --- | ---
$$5$$ | 11 | 3.4 | 0.56 | 0.19
$$500$$ | 12 | 5.0 | 2.5 | 1.6
$$1000$$ | 13 | 7.7 | 4.3 | 3.4
$$2000$$ | 13 | 16 | 14 | 13

Here the story is similar to above: for small dimensionality the full Jax sampler is 60x faster than the Python loop version. But as you increase the dimension the gap gets smaller. As in the previous case, this is probably because the main effect of increasing the dimensionality is seen in the log-posterior function (which is in Jax for all the samplers).

The only difference to note is that the Jax kernel version is slower than the Python loop version for dimension 2000. I also don't have explanation for this! This reinforces the idea that you should always benchmark your code to make sure it's fast.

## Stochastic gradient Langevin dynamics (SGLD)

We now try the same experiment with [stochastic gradient langevin dynamics](https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics) sampler. This is the same as ULA but calculates gradients based on mini-batches rather than on the full dataset. This makes it suited for application with very large datasets, but the sampler produces samples that aren't exactly from the target distribution (often the variance is too high).

The transition kernel below is therefore quite similar to ULA, but randomly chooses minibatches of data to calculate gradients with. Note also that the `grad_log_post` function includes the minibatch dataset as arguments. Also note that we sample minibatches _with_ replacement (`random.choice` has `replace=True` as default). This is because sampling without replacement is very expensive is Jax, so doing this will dramatically slow down the sampler!

```python
@partial(jit, static_argnums=(2,3,4,5,6))
def sgld_kernel(key, param, grad_log_post, dt, X, y_data, minibatch_size):
    N, _ = X.shape
    key, subkey1, subkey2 = random.split(key, 3)
    idx_batch = random.choice(subkey1, N, shape=(minibatch_size,))
    paramGrad = grad_log_post(param, X[idx_batch], y_data[idx_batch])
    param = param + dt*paramGrad + np.sqrt(2*dt)*random.normal(key=subkey2, shape=(param.shape))
    return key, param
```

### Increase amount of data:

We run the same experiment as before: `20 000` samples for a 5 dimensional parameter. with increasing the amount of data. As before the timings are in seconds. As minibatch size we use 10% of the dataset for first two datasets, and 1% of the datasets for the last two.

dataset size | python | Jax kernel | full Jax (1st run) | full Jax (1nd run)
--- | --- | --- | --- | ---
$$10^3$$ | 59 | 4.2 | 1.1 | 0.056
$$10^4$$ | 60 | 4.4 | 3.8 | 0.9
$$10^5$$ | 73 | 3.9 | 1.6 | 0.40
$$10^6$$ | 65 | 4.0 | 2.0 | 0.69


Here we see that unlike in the case of ULA, we keep the large speedup from using compiling everything in Jax. This is because the minibatches allow us to keep the cost of the log-posterior low.

We also notice that the Python and Jax kernel versions are slower than their ULA counterparts for low and medium datset sizes. This is probably due to the cost of sampling the minibatches and that for these small dataset sizes the log-posterior function is efficient enough to not need minibatches. However for the last dataset (1 million data points) the benefits of using minibatches becomes clear.


### Increase the dimension:

We now run the samplers with a fixed dataset size of 1000 data points, and run each sampler for 20K iterations while varying the dimension. We use as minibatch size 10% of the data for all 4 runs.


dimension | python | Jax kernel | full Jax (1st run) | full Jax (1nd run)
--- | --- | --- | --- | ---
$$5$$ |  61 | 4.2 | 1.1 | 0.055
$$500$$ |  62 | 4.2 | 1.9 | 0.56
$$1000$$ |  62 | 5.0 | 2.3 | 0.98
$$2000$$ |  68 | 6.4 | 3.3 | 1.95


Here the two Jax samplers benefit from using minibatches, while the Python version is slower than its ULA counterpart in all cases.


## Metropolis Adjusted Langevin algorithm (MALA)

We now re-run the same experiment but with [MALA](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm), which is like ULA but with a Metropolis-Hastings correction to ensure that the samples are unbiased. This correction means that the transition kernel is more computationally expensive:


```python
@partial(jit, static_argnums=(3,5))
def mala_kernel(key, paramCurrent, paramGradCurrent, log_post, logpostCurrent, dt):
    key, subkey1, subkey2 = random.split(key, 3)
    paramProp = paramCurrent + dt*paramGradCurrent + np.sqrt(2*dt)*random.normal(key=subkey1, shape=paramCurrent.shape)
    new_log_post, new_grad = log_post(paramProp)

    term1 = paramProp - paramCurrent - dt*paramGradCurrent
    term2 = paramCurrent - paramProp - dt*new_grad
    q_new = -0.25*(1/dt)*np.dot(term1, term1)
    q_current = -0.25*(1/dt)*np.dot(term2, term2)

    log_ratio = new_log_post - logpostCurrent + q_current - q_new
    acceptBool = np.log(random.uniform(key=subkey2)) < log_ratio
    paramCurrent = np.where(acceptBool, paramProp, paramCurrent)
    current_grad = np.where(acceptBool, new_grad, paramGradCurrent)
    current_log_post = np.where(acceptBool, new_log_post, logpostCurrent)
    accepts_add = np.where(acceptBool, 1,0)
    return key, paramCurrent, current_grad, current_log_post, accepts_add
```


We run the usual 3 versions: a Python sampler with Jax for the log-posterior, a Python loop with the Jax transition kernel, and a "full Jax" sampler.

### Increase amount of data:

We run each sampler for `20 000` samples for a 5 dimensional parameter while varying the size of the dataset.

dataset size | python | Jax kernel | full Jax (1st run) | full Jax (2nd run)
--- | --- | --- | --- | ---
$$10^3$$ | 38 | 7 | 0.93 | 0.19
$$10^4$$ | 38 | 7 | 2.6 | 1.9
$$10^5$$ | 56 | 35 | 27 | 26
$$10^6$$ | 330 | 310 | 270 | 272

The story here is similar to the story in the case of ULA. The main difference is that the speedup for the full Jax sampler is more pronounced in this case (especially for the smaller datasets). Indeed, for 1000 data points the full Jax (once it's compiled) is 200 times faster than the Python loop version. This is probably because the transition kernel is more complicated and so contributes more to the overall computational cost of the sampler. As a result compiling it brings a larger speed increase than for ULA.

Furthermore (in the case of ULA) when the dataset size is increased the speed of the samplers start to converge to the same value.

### Increase the dimension:

We now run the samplers with a fixed dataset size of 1000 data points, and run each sampler for 20K iterations while varying the dimension.


dimension | python | Jax kernel | full Jax (1st run) | full Jax (1nd run)
--- | --- | --- | --- | ---
$$5$$ | 39 | 7.2 | 0.94 | 0.20
$$500$$ | 40 | 7.2 | 2.6 | 1.5
$$1000$$ | 41 | 8.0 | 4.8 | 3.4
$$2000$$ | 43 | 15 | 14 | 13

We we have a similar story to the case of increasing data: using full Jax speeds up the sampler a lot, but that gap gets smaller as you increase the dimensionality.


# Conclusion

We've seen that there are different ways to write MCMC samplers by having more or less of the code written in Jax. On one hand, you can use Jax to write the log-posterior function and use Python/NumPy for the rest. On the other hand you can use Jax to write the entire sampler. We've also seen that in general the full Jax sampler is faster than the Python loop version, but that this difference gets smaller as the amount of data and dimensionality increases.

The main conclusion we take from this is that in general writing more things in Jax speeds up the code. However you have to make sure it's well written so you don't accidentally slow things down (for example by re-compiling a function at every iteration by mis-using `static_argnums` when jitting). You should therefore always benchmark code and compare different ways of writing it.
