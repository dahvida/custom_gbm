from jax import config
config.update("jax_enable_x64", True)
import lightgbm as lgb
import numpy as np
from scipy.special import expit

###############################################################################

class CustomGBM:

    def __init__(
            self,
            loss_fn,
            loss_params,
            booster_params
            ):
                
        self.loss_fn = loss_fn
        self.loss_params = loss_params
        self.booster_params = booster_params
        self.booster = None
    
    def build_loss(
            self,
            y_train,
            ):
        
        params = self.loss_params.copy()
        self.loss = self.loss_fn(y_train, **params)
        self.loss.get_init_score()

    def fit(
            self,
            x,
            y
            ):
        
        self.build_loss(y)
        
        train_dataset = lgb.Dataset(
                    x, y, init_score=np.full_like(y, self.loss.init_score, dtype=np.float64)
                    )
        
        params = self.booster_params.copy()
        num_boost_round = params["num_boost_round"]
        
        self.booster = lgb.train(
                params = params,
                train_set = train_dataset,
                fobj = self.loss.get_gradient,
                feval = self.loss.get_metric,
                num_boost_round = num_boost_round,
                  )

    def predict(
            self,
            x
            ):
        return expit(self.booster.predict(x) + self.loss.init_score)


