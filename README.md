# eDNA JOINT SPECIES DISTRIBUTION MODEL

`eDNA data preparation & Bayesian jSDM implementation.Rmd` documents the workflow used to prepare eDNA data, fit spatially blocked joint species distribution models using boral, evaluate marginal predictions, and generate posterior community realisations for the best-of-N analysis.

- The script requires the R packages listed in its configuration section. It does not install packages automatically.
- Sample-level eDNA data and environmental covariates are not distributed with this repository.
- The 50-km aggregated dataset is provided separately for data sharing and reuse (DOI: 10.5061/dryad.cnp5hqcn3). It cannot be used to rerun the analyses in `eDNA data preparation & Bayesian jSDM implementation.Rmd`.
  
The eDNA script is therefore provided as transparent analytical documentation rather than as an executable reproducibility package.

# TRAINING MLP AND AR MODELS
- Install the required dependencies using `python -m pip install -r requirements.txt`
- Use `training.py`  to train the marignal and autoregressive models.
- `torchrun` can be used for distributed learning.
- Set the configuration parameters in `config.cfg file`. Example settings are already provided.

# EVALUATING MODELS
- Specify the models to evaluate in the "Evaluated Instances" section of the `config.cfg`file. Example configurations for all three model types are provided.
- Run `evaluation.py`. `torchrun`can also be used for distributed evaluation.
- Plots are saved as `fi_curve.png` in the main directory.
