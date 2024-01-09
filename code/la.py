from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
from scipy import optimize as opt
from typing import *
from loss_function import loss_function

###############################################################################

class LA_loss(loss_function):
    "based on https://github.com/google-research/google-research/tree/master/logit_adjustment"
    def __init__(self,
                 y_true,
                 tau = 1.0,
                 epsilon = 0.0,
                 class_weight = None):
        
        super(LA_loss, self).__init__(y_true, class_weight)
        
        self.tau = jnp.array(tau)
        self.epsilon = jnp.array(epsilon)

        pi_pos = self.n_minority / (self.n_minority + self.n_majority)
        pi_neg = self.n_majority / (self.n_minority + self.n_majority)
        scale = pi_pos * y_true + pi_neg * (1 - y_true)
        self.shift = jnp.log(scale**self.tau + 1e-12)

    @partial(jit, static_argnums=(0,))
    def __call__(self, y_true, y_pred):
                
        y_pred = y_pred + self.shift
        
        p = 1/(1+jnp.exp(-y_pred))
        q = 1-p
        
        pos_loss = (jnp.log(p) + self.epsilon * (1 - p))* self.alpha_m
        neg_loss = (jnp.log(q) + self.epsilon * (1 - q))* self.alpha_M
        
        return y_true * pos_loss + (1 - y_true) * neg_loss
