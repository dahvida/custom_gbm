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

class Focal_loss(loss_function):
    "based on https://github.com/artemmavrin/focal-loss/tree/7a1810a968051b6acfedf2052123eb76ba3128c4"
    def __init__(self,
                 y_true,
                 gamma = 2.0,
                 epsilon = 0.0,
                 class_weight=None):
        
        super(Focal_loss, self).__init__(y_true, class_weight)
        self.gamma = jnp.array(gamma)
        self.gamma_d = gamma + 1.0
        self.epsilon = jnp.array(epsilon)

    @partial(jit, static_argnums=(0,))
    def __call__(self, y_true, y_pred):
        
        p = 1/(1+jnp.exp(-y_pred))
        q = 1-p
        
        focal_pos = jnp.power(q, self.gamma) * jnp.log(p)
        poly1_pos = self.epsilon * jnp.power(q,self.gamma_d)
        pos_loss = jnp.add(focal_pos, poly1_pos) * self.alpha_m
        
        focal_neg = jnp.power(p, self.gamma) * jnp.log(q)
        poly1_neg = self.epsilon * jnp.power(p,self.gamma_d)
        neg_loss = jnp.add(focal_neg, poly1_neg) * self.alpha_M
        
        return y_true * pos_loss + (1 - y_true) * neg_loss
