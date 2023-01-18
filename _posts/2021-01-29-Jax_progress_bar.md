---
layout: post
title: "How to add a progress bar to JAX scans and loops"
date: 2021-01-29 08:00:00 +0000
categories: JAX programming MCMC statistics
---


JAX allows you to write optimisers and samplers which are really fast if you use the [`scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) or [`fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) functions. However if you write them in this way it's not obvious how to add progress bar for your algorithm. This post explains how to make a progress bar using Python's `print` function as well as using [tqdm](https://pypi.org/project/tqdm/). After briefly setting up the sampler, we first go over how to create a basic version using Python's `print` function, and then show how to create a nicer version using tqdm. You can find the code for the basic version [here](https://gist.github.com/jeremiecoullon/4ae89676e650370936200ec04a4e3bef) and the code for the tqdm version [here](https://gist.github.com/jeremiecoullon/f6a658be4c98f8a7fd1710418cca0856).

_Update January 2023: this is now available in a pip-installable package: [JAX-tqdm](https://github.com/jeremiecoullon/jax-tqdm)_


# Setup: sampling a Gaussian

We'll use an [Unadjusted Langevin Algorithm](https://en.wikipedia.org/wiki/Langevin_dynamics) (ULA) to sample from a Gaussian to illustrate how to write the progress bar. Let's start by defining the log-posterior of a d-dimensional Gaussian and we'll use JAX to get it's gradient:


```python
@jit
def log_posterior(x):
    return -0.5*jnp.dot(x,x)

grad_log_post = jit(grad(log_posterior))
```

We now define ULA using the `scan` function (see [this post]({% post_url 2020-10-18-MCMCJax3ways %}) for an explanation of the `scan` function).


```python
@partial(jit, static_argnums=(2,))
def ula_kernel(key, param, grad_log_post, dt):
    key, subkey = random.split(key)
    paramGrad = grad_log_post(param)
    noise_term  = jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    param = param + dt*paramGrad + noise_term
    return key, param


@partial(jit, static_argnums=(1,2,))
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

# Basic progress bar


As a workaround, the JAX team has added the [`host_callback`](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html) module (which is still experimental, so things may change). This module defines functions that allow you to call Python functions from within a JAX function. Here's how you would use the [`id_tap`](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html#using-id-tap-to-call-a-jax-function-on-another-device-with-no-returned-values-but-full-jax-transformation-support) function to create a progress bar (from this [discussion](https://github.com/google/jax/discussions/4763#discussioncomment-121452)):

```python
from jax.experimental import host_callback

def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(f"Iteration {iter_num:,} / {num_samples:,}")

@jit
def progress_bar(arg, result):
    """
    Print progress of a scan/loop only if the iteration number is a multiple of the print_rate

    Usage: `carry = progress_bar((iter_num + 1, num_samples, print_rate), carry)`
    Pass in `iter_num + 1` so that counting starts at 1 and ends at `num_samples`

    """
    iter_num, num_samples, print_rate = arg
    result = lax.cond(
        iter_num % print_rate==0,
        lambda _: host_callback.id_tap(_print_consumer, (iter_num, num_samples), result=result),
        lambda _: result,
        operand=None)
    return result
```

The `id_tap` function behaves like the identity function, so calling `host_callback.id_tap(_print_consumer, (iter_num, num_samples), result=result)` will simply return `result`. However while doing this, it will also call the function `_print_consumer((iter_num, num_samples))` which we've defined to print the iteration number.

You need to pass an argument in this way because you need to include a data dependency to make sure that the print function gets called at the correct time. This is linked to the fact that computations in JAX are run [only when needed](https://jax.readthedocs.io/en/latest/async_dispatch.html). So you need to pass in a variable that changes throughout the algorithm such as the PRNG key at that iteration.

Also note also that the `_print_consumer` function takes in `arg` (which holds the current iteration number as well as the total number of iterations) and `transform`. This `transform` argument isn't used here, but apparently should be included in the consumer for id_tap (namely: the Python function that gets called).

Here's how you would use the progress bar in the ULA sampler:

```python  
def ula_step(carry, iter_num):
    key, param = carry
    key = progress_bar((iter_num + 1, num_samples, print_rate), key)
    key, param = ula_kernel(key, param, grad_log_post, dt)
    return (key, param), param
```

We passed the `key` into the progress bar which comes out unchanged. We also set the print rate to be 10% of the number of samples. Note that this would also work for [`lax.fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) except that the first argument of `ula_step` would be the current iteration number.


### Put it in a decorator

We can make this even easier to use by putting the progress bar in a decorator. Note that the decorator takes in `num_samples` as an argument.

```python
def progress_bar_scan(num_samples):
    def _progress_bar_scan(func):
        print_rate = int(num_samples/10)
        def wrapper_progress_bar(carry, iter_num):
            iter_num = progress_bar((iter_num + 1, num_samples, print_rate), iter_num)
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

Now that we have a progress bar, we might also want to know when the function is compiling (which is especially useful when it takes a while to compile). Here we can use the fact that the `print` function only gets called during compilation. We can add `print("Compiling..")` at the beginning of `ula_sampler_pbar` and add `print("Running:")` at the end. Both of these will then only display when the function is first run. You can find the code for this sampler [here](https://gist.github.com/jeremiecoullon/4ae89676e650370936200ec04a4e3bef).


# tqdm progress bar

We'll now use the same ideas to build a fancier progress bar: namely one that uses [tqdm](https://pypi.org/project/tqdm/). We'll need to use `host_callback.id_tap` to define a `tqdm` progress bar and then call `tqdm.update` regularly to update it. We'll also need to close the progress bar once we're finished or else `tqdm` will act weirdly. To do with we'll define a decorator that takes in arguments just like we did in the case of the simple progress bar.

This decorator defines the tqdm progress bar at the first iteration, updates it every `print_rate` number of iterations, and finally closes it at the end. You can optionally pass in a message to add at the beginning of the progress bar.

There are details to make sure the progress bar acts correctly in corner cases, such as if `num_samples` is less than 20, or if it's not a multiple of 20. Note also that tqdm is closed at the last iteration only _after_ the parameter update is done.

```python
def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
            message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1 # if you run the sampler for less than 20 iterations
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )


    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan

```

Although this progress bar is more complicated than the previous one, you use it in exactly the same way. You simply add the decorator to the step function used in `lax.scan` with the number of samples as argument (and optionally the messsage to print at the beginning of the progress bar).

```python
@partial(jit, static_argnums=(1,2))
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

### Conclusion

So we've built two progress bars: a basic version and a nicer version that uses tqdm. The code for these are on these two gists: [here](https://gist.github.com/jeremiecoullon/4ae89676e650370936200ec04a4e3bef) and [here](https://gist.github.com/jeremiecoullon/f6a658be4c98f8a7fd1710418cca0856).
