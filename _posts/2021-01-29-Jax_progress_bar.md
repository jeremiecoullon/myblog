---
layout: post
title: "How to add a progress bar to JAX loops and scans"
date: 2021-01-29 08:00:00 +0000
categories: JAX programming MCMC statistics
---


JAX allows you to write optimisers and samplers which are especially fast if you use the [`scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) or [`fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) functions. However if you write them in this way it's not obvious how to add progress bar for your algorithm (using [tqdm](https://pypi.org/project/tqdm/) or simply the `print` function for example). This post explains how to do it. You can find all the code in this [Github gist](https://gist.github.com/jeremiecoullon/4ae89676e650370936200ec04a4e3bef).


### Setup: sampling a Gaussian

We'll use an [Unadjusted Langevin Algorithm](https://en.wikipedia.org/wiki/Langevin_dynamics) (ULA) to sample from a Gaussian to illustrate how to write the progress bar. Let's start by defining the log-posterior of a d-dimensional Gaussian and we'll use JAX to get it's gradient:


```python
@jit
def log_posterior(x):
    return -0.5*jnp.dot(x,x)

grad_log_post = jit(grad(log_posterior))
```

We now define ULA using the `scan` function (see [this post]({% post_url 2020-10-18-MCMCJax3ways %}) for an explanation of the `scan` function).


```python
@partial(jit, static_argnums=(2,3))
def ula_kernel(key, param, grad_log_post, dt):
    key, subkey = random.split(key)
    paramGrad = grad_log_post(param)
    noise_term  = jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    param = param + dt*paramGrad + noise_term
    return key, param


@partial(jit, static_argnums=(1,2,3))
def ula_sampler(key, grad_log_post, num_samples, dt, x_0):

    def ula_step(carry, iter_num):
        key, param = carry
        key, param = ula_kernel(key, param, grad_log_post, dt)
        return (key, param), param

    carry = (key, x_0)
    _, samples = lax.scan(ula_step, carry, jnp.arange(num_samples))
    return samples
```

If we add a `print` function in `ula_step` above, it will only be called the first time it is called, which is when `ula_sampler` is compiled. This is because printing is a side effect, and [compiled JAX functions are pure](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Pure-functions).

### Build a progress bar using `host_callback.id_tap`

As a workaround, the JAX team has added the [`host_callback`](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html) module (which is still experimental, so things may change). This module defines functions that allow you to call Python functions from within a JAX function. Here's how you would use the [`id_tap`](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html#using-id-tap-to-call-a-jax-function-on-another-device-with-no-returned-values-but-full-jax-transformation-support) function to create a progress bar (from this [discussion](https://github.com/google/jax/discussions/4763#discussioncomment-121452)):

```python
from jax.experimental import host_callback

def _print_consumer(arg, transform):
    iter_num, n_iter = arg
    print(f"Iteration {iter_num}/{n_iter}")

@jit
def progress_bar(arg, result):
    """
    Print progress of a scan/loop only if the iteration number is a multiple of the print_rate

    Usage: carry = progress_bar((iter_num, n_iter, print_rate), carry)
    """
    iter_num, n_iter, print_rate = arg
    result = lax.cond(
        i%print_rate==0,
        lambda _: host_callback.id_tap(_print_consumer, (iter_num, n_iter), result=result),
        lambda _: result,
        operand=None)
    return result
```

The `id_tap` function behaves like the identity function, so calling `host_callback.id_tap(_print_consumer, (iter_num, n_iter), result=result)` will simply return `result`. However while doing this, it will also call the function `_print_consumer((iter_num, n_iter))` which we've defined to print the iteration number.

You need to pass an argument in this way because you need to include a data dependency to make sure that the print function gets called at the correct time. This is linked to the fact that computations in JAX are run [only when needed](https://jax.readthedocs.io/en/latest/async_dispatch.html). So you need to pass in a variable that changes throughout the algorithm such as the PRNG key at that iteration.

Also note also that the `_print_consumer` function takes in `arg` (which holds the current iteration number as well as the total number of iterations) and `transform`. This `transform` argument isn't used here, but apparently should be included in the consumer for id_tap (namely: the Python function that gets called).

Here's how you would use the progress bar in the ULA sampler:

```python  
  def ula_step(carry, iter_num):
      key, param = carry
      key = progress_bar((iter_num, num_samples, print_rate), key)
      key, param = ula_kernel(key, param, grad_log_post, dt)
      return (key, param), param
```

We passed the `key` into the progress bar which comes out unchanged. We also set the print rate to be 10% of the number of samples.


### Put it in a decorator

We can make this even easier to use by putting the progress bar in a decorator. Note that the decorator takes in `num_samples` as an argument.

```python
def progress_bar_scan(num_samples):
    def _progress_bar_scan(func):
        print_rate = int(num_samples/10)
        def wrapper_progress_bar(carry, iter_num):
            iter_num = progress_bar((iter_num, num_samples, print_rate), iter_num)
            return func(carry, iter_num)
        return wrapper_progress_bar
    return _progress_bar_scan
```

Remember that writing a decorator with arguments means writing a function that returns a decorator (which itself is a function that returns a modified version of the main function you care about). See this [StackOverflow question](https://stackoverflow.com/questions/5929107/decorators-with-parameters) about this.

Putting it all together, the result is very easy to use:

```python
@partial(jit, static_argnums=(1,2,3))
def ula_sampler_pbar(key, grad_log_post, num_samples, dt, x_0):
    "ULA sampler with progress bar"

    @progress_bar_scan(num_samples)
    def ula_step(carry, iter_num):
        key, param = carry
        key, param = ula_kernel(key, param, grad_log_post, dt)
        return (key, param), param

    carry = (key, x_0)
    _, samples = lax.scan(ula_step, carry, jnp.arange(num_samples))
    return samples
```


### A final touch: use `print` to see when the function is compiling

Now that we have a progress bar, we might also want to know when the function is compiling (which is especially useful when it takes a while to compile). Here we can use the fact that the `print` function only gets called during compilation. We can add `print("Compiling..")` at the beginning of `ula_sampler_pbar` and add `print("Running:")` at the end. Both of these will then only display when the function is first run.

### Conclusion

That's it! You can try out this code in this [Github gist](https://gist.github.com/jeremiecoullon/4ae89676e650370936200ec04a4e3bef).
