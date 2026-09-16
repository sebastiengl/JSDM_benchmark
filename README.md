# TRAINING MLP AND AR MODELS
- Install required dependencies using `python -m pip install -r requirements.txt`
- Use the `training.py` script for the training of the marignal and autoregressive models.
- `torchrun` can be used for distributed learning.
- Add your configuration parameters in the `config.cfg file`. For a simple example, all parameters should be already configurated.

# EVALUATION MODELS
- Add the models you evaluated in "Evaluated Instances" of the `config.cfg`file. All 3 types of models are already in example for configuration.
- Run the `evaluation.py` script. `torchrun`can also be used in this case.
- Plot should be saved under the `fi_curve.png` in the main directory.
