# Custom_gbm
JAX-based implementation of custom loss functions for LightGBM. The package currently includes:

- **Focal loss** (https://arxiv.org/abs/1708.02002)  
- **LDAM loss** (https://arxiv.org/abs/1906.07413)  
- **Logit-Adjusted loss** (https://arxiv.org/abs/2007.07314)  
- **PolyLoss** (https://arxiv.org/abs/2204.12511)  

More will come in the future. Additionally, the package provides a wrapper for facilitating the addition of new loss functions, e.g. by handling automatic differentiation, initial score calculation, interfacing with LightGBM Train API and so forth.   

# Installation  
The package requires LightGBM 3.5.5, Scipy 1.10.1 and Jax 0.4.12. Alternatively, you can create a virtual environment using the `environment.yml` file in the Github repository.   
```
git clone https://github.com/dahvida/custom_gbm
conda env create --name customgbm --file=environment.yml
conda activate customgbm
```

# Usage
Here is a minimal example for training a model with CustomGBM. Additional tutorials are provided in the [notebook](notebook) folder.
```python
# import packages
from focal import Focal_loss
from custom_gbm import CustomGBM

# define parameters
loss_fn = Focal_loss
loss_params = {"gamma": 2.0}
booster_params = {"num_boost_round": 100}
gbm = CustomGBM(loss_fn, loss_params, booster_params)

# fit model
gbm.fit(x_train, y_train)

# get predictions
predictions = gbm.predict(x_test)
```
