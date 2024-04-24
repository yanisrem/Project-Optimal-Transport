from typing import Any, Tuple
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
from jax import jit
import optax
from optax._src import base
import flax.linen as nn
from flax.training import train_state

import ott
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from src.icnn import ICNN

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class OT_Solver:
    """Compute OT map"""

    def __init__(self,
                 input_dim: int,
                 neural_net: Optional[nn.Module] = None,
                 optimizer: Optional[base.GradientTransformation] = None,
                 num_train_iters: int = 20000,
                 plot_function = None,
                 seed: int = 0):

        self.num_train_iters = num_train_iters
        self.plot_function = plot_function

        # set random key
        rng = jax.random.PRNGKey(seed)
        rng, rng_setup = jax.random.split(rng,2)
        self.key = rng

        # set default optimizer
        if optimizer is None:
            optimizer = optax.adam(learning_rate=0.001)

        # set default neural architecture
        if neural_net is None:
            neural_net = ICNN(dim_hidden=[64, 64, 64])

        self.setup(rng_setup, neural_net, input_dim, optimizer)


    def setup(self, rng, neural_net, input_dim, optimizer):
        """Setup all components for training"""
        rng, rn_state = jax.random.split(rng,2)
        self.state = self.create_train_state(rn_state, neural_net, optimizer, input_dim) # contains model, parameters and optimizer
        self.train_step = self.get_step_fn()


    def __call__(self,
                 sampler_source: Iterator[jnp.ndarray],
                 sampler_target: Iterator[jnp.ndarray],
                 size_batch_train):
        _ = self.train_quantile_net(sampler_source, sampler_target, size_batch_train)
        return self.state

    def train_quantile_net(self,sampler_source, sampler_target, size_batch_train):
        batch = {}
        master_key = self.key
        for step in range(self.num_train_iters):
            # generate new batch
            master_key, inner_key_source, inner_key_target = jax.random.split(master_key, num=3)
            batch['source'] = sampler_source.generate_samples(inner_key_source, size_batch_train)
            batch['target'] = sampler_target.generate_samples(inner_key_target, size_batch_train)

            # compute the loss function and apply a gradient step
            self.state, loss = self.train_step(self.state, batch)

            if step % 100 == 0:
                print('loss After {} Iterations : {:.6f}'.format(step, loss))

            # plot some figures
            if step % 1000 == 0 and self.plot_function:
                master_key, inner_key_source, inner_key_target = jax.random.split(master_key, num=3)
                X_train = sampler_source.generate_samples(inner_key_source,size_batch_train)
                Y_train = sampler_target.generate_samples(inner_key_target,size_batch_train)
                self.plot_function(X_train, Y_train, self.state)
        return None

    def create_train_state(self, rng, neural_net, optimizer, input_dim):
        params = neural_net.init(rng, jnp.ones(input_dim))['params']
        return train_state.TrainState.create(apply_fn=neural_net.apply, params=params, tx=optimizer)

    def get_step_fn(self):
        def loss_fn(params, predict, batch):
            """This is the loss function it should return a number

            Hint: it should follow those steps:
              - compute the gradient of the ICNN with respect to the input to get the ot map
              - use vmap so that ot_map take as input batch of samples
              - transport the source samples
              - Minimize the distance between the transported samples and the target samples
                using sinkhorn_divergence
            """

            source, target = batch['source'], batch['target']
            # compute the gradient of the ICNN wrt the input
            ot_map_point = jax.grad(predict, argnums=1)
            # use vmap so that ot_map_point take as input batch of samples
            ot_map = jax.vmap(lambda x: ot_map_point({'params': params}, x))
            # Transport the source samples
            predicted = ot_map(source)
            # Minimize the distance between the transported samples and the target samples
            loss = sinkhorn_divergence(pointcloud.PointCloud, predicted, target,
                            relative_epsilon=True).divergence
            return loss

        @jax.jit
        def step_fn(state, batch):
            value_and_grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = value_and_grad_fn(state.params, state.apply_fn, batch)
            return state.apply_gradients(grads=grad), loss
        return step_fn