# GDESA: Greeedy Diversity Encoder with Self-Attention for Search Result Diversification

### Instructions
Trained models and baseline runs are listed in `models/` and `baselines/`.



### Data Preparation
GDESA is based on the same preprocessed data as DSSA. You can download and decompress `data_cv.tar.gz` from the repo of [DSSA](https://github.com/jzbjyb/DSSA). Notice that the `data` folder in DSSA is also required.

The list-pairwise training samples should be deployed as compressed pickles, use `data_pickle.py` to do this.

### Dependencies
See `requirements.txt` for more details. The requirements of GDESA is almost the same with DSSA, while `tensorflow` is replaced with `torch` and `torchtext`.

### Reproduce Experiments
Run `infer_reproduce.py` to reproduce the 5-fold cross validation based on 5 different models. The ranking results will be written into `result.json`

We will soon update the full instructions of model training.