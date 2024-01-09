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

class Poly_loss(loss_function):
    "based on https://github.com/kaidic/LDAM-DRW"
    def __init__(self,
                 y_true,
                 epsilon = 0.0,
                 class_weight = None):
        
        super(Poly_loss, self).__init__(y_true, class_weight)
        self.epsilon = jnp.array(epsilon)
    
    @partial(jit, static_argnums=(0,))
    def __call__(self, y_true, y_pred):

        p = 1/(1+jnp.exp(-y_pred))
        q = 1-p
        pos_loss = (jnp.log(p) + self.epsilon * (1 - p)) * self.alpha_m
        neg_loss = (jnp.log(q) + self.epsilon * (1 - q)) * self.alpha_M
        
        return y_true * pos_loss + (1 - y_true) * neg_loss 
