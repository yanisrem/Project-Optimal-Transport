from typing import Any, Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jit
from functools import partial

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class PositiveDense(nn.Module):
    """A linear transformation using a non-negative matrice of weights"""
    dim_hidden: int
    beta: int = 30.0
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))
        kernel = 1/self.beta * nn.softplus( self.beta * kernel)
        x = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        return x
    

class ICNN(nn.Module):
  """ A fully connected input convex neural network """
  dim_hidden: Sequence[int]
  act_fn: Callable = nn.elu
  init_fn: Callable = nn.initializers.variance_scaling(scale=1., distribution="normal", mode="fan_avg")

  def setup(self):
    self.dim = self.dim_hidden + (1,)
    self.w_zs = [PositiveDense(feature, kernel_init=self.init_fn) for feature in self.dim]
    self.w_ys = [nn.Dense(feature, use_bias=True, kernel_init=self.init_fn) for feature in self.dim]

  @nn.compact
  def __call__(self, input):
    y = input
    z = jnp.zeros(shape=y.shape)
    for w_z, w_y in zip(self.w_zs[:-1], self.w_ys[:-1]):
      z = self.act_fn(w_z(z) + w_y(y))
    z = self.w_zs[-1](z) + self.w_ys[-1](y)
    return z.squeeze()

class sampler_from_data:
    def __init__(self, x ):
        self.x = x
        self.setup()

    def setup(self):
        @partial(jit, static_argnums=1)
        def generate_samples(key, num_samples):
            points = jax.random.choice(key, self.x, (num_samples,))
            return points

        # define samples generator
        self.generate_samples = generate_samples
