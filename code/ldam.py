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

class LDAM_loss(loss_function):
    "based on https://github.com/kaidic/LDAM-DRW"
    def __init__(self,
                 y_true,
                 max_m = 0.5,
                 epsilon = 0.0,
                 class_weight = None):
        
        super(LDAM_loss, self).__init__(y_true, class_weight)
        cls_num_list = jnp.array([self.n_majority, self.n_minority])
        m_list = 1.0 / jnp.sqrt(jnp.sqrt(cls_num_list))
        m_list = m_list * (max_m / jnp.max(m_list))
        self.m_list = m_list
        self.epsilon = jnp.array(epsilon)
        
        self.shift = y_true.copy()
        self.shift[jnp.where(self.shift==0)] = self.m_list[0]
        self.shift[jnp.where(self.shift==1)] = self.m_list[1]
    
    @partial(jit, static_argnums=(0,))
    def __call__(self, y_true, y_pred):
        
        y_m = y_pred - self.shift

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = y_true > 0
        output = jnp.where(index_bool, y_m, y_pred)

        p = 1/(1+jnp.exp(-output))
        q = 1-p
        pos_loss = (jnp.log(p) + self.epsilon * (1 - p)) * self.alpha_m
        neg_loss = (jnp.log(q) + self.epsilon * (1 - q)) * self.alpha_M
        
        return y_true * pos_loss + (1 - y_true) * neg_loss
