from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
from scipy import optimize as opt
from typing import *

###############################################################################

class loss_function:
    
    def __init__(self, 
                 y_true, 
                 class_weight):
        
        self.e = jnp.array(1e-3, dtype=jnp.float64)
        
        self.y_true = y_true        
        self.n_majority = len(y_true) - jnp.sum(y_true)
        self.n_minority = jnp.sum(y_true)
        
        majority_ratio = 1 - (self.n_majority / (self.n_majority + self.n_minority))
        minority_ratio = 1 - (self.n_minority / (self.n_majority + self.n_minority))
        
        if class_weight == "balanced":
            self.alpha_m = minority_ratio
            self.alpha_M = majority_ratio
        elif class_weight == "sqrt":
            self.alpha_m = jnp.sqrt(minority_ratio)
            self.alpha_M = jnp.sqrt(majority_ratio)
        elif class_weight == "div5":
            self.alpha_m = minority_ratio / 5
            self.alpha_M = majority_ratio / 5
        elif class_weight == None:
            self.alpha_m = 1.0
            self.alpha_M = 1.0
        else:
            self.alpha_m = class_weight[0]
            self.alpha_M = class_weight[1]
            
        self.init_score = 0    
                
    def get_init_score(self):
        res = opt.minimize_scalar(
            lambda p: -self.__call__(self.y_true, p).sum(),
            bounds=(-10, 10),
            method='bounded')
        self.init_score = np.array(res.x)
    
    def get_metric(self,
                   y_pred,
                   y_true):

        y_true = y_true.get_label()
        loss = -jnp.mean(self.__call__(y_true, y_pred))
        
        return "loss", np.array(loss), False
    
    def get_gradient(self,
                     y_pred,
                     y_true
                     ) -> Tuple[np.ndarray, np.ndarray]:
        
        y_true = y_true.get_label()        
        grad, hess = self.numerical_derivatives(y_pred, y_true)
        
        return np.array(grad), np.array(hess)
    
    @partial(jit, static_argnums=(0,))
    def numerical_derivatives(self,
                              y_pred,
                              y_true):
        
        loss = self.__call__(y_true, y_pred)
        diff1 = self.__call__(y_true, y_pred + self.e)
        diff2 = self.__call__(y_true, y_pred - self.e)
        
        grad = - (diff1 - diff2) / (2*self.e)
        hess = - (diff1 - 2*loss + diff2) / self.e**2
        
        return grad, hess

