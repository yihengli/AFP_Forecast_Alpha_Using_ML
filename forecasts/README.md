## AFP: Forecasting Scaffold

### How to replicate the environment

If you are using `pip`, simply run `pip install -r requirements.txt` under your new `virtualenv`.

If you are using `conda`, simply run `conda env create -f env_mac.yml` or `env_win.yml` based on your system.

If there are some problems to install specific pacakges, try to install them manually.

If there are some pacakges missing when running the codes, try to install them manually.

### How to use the forecast scaffold

Once the environment is set up, use the `main.py` to run the scaffold.

You can also run `python main.py --help` to see the helpers for each argument.

Currently only supports `forecast` command, you can check its details by using `python main.py forecast --help`.

#### Rolling Regression

```
python3 main.py --name your_task_name \
                forecast \
                --model Regression \
                --rolling \
                --rolling-bars 252 \
                --predict-bars 20 \
                --minimum-train-bars 252 \
                --freq d \
                --train-periods 2001-01-01 2014-12-31 \
                --test-periods 2015-01-01 2018-12-31 \
                --lags 1 \
                --forward-bars 3
```

* `--name`: The name for this task, any outputs files will saved by this name
* `--model`: By specifying `Regression`, you are using `statsmodels.OLS` model without intercept
* `--rolling`: By giving this argument, the model will be evaluated at a rolling basis.
* `--rolling-bars`: Every time uses this number of bars to train the regression mdoel
* `--predict-bars`: Once trained, use the trained model to make this number of predictions going forward.
* `--minimum-train-bars`: Check if the dataset has at least this number of data points, otherwise ignore this dataset.
* `--freq`: If specified, data will be resampled to given frequency, (e.g. `w-Fri`, `m`, `a`, etc.) 
* `--train-periods`: inclusively defines in-sample periods, it is not meaningful for rolling method
* `--test-periods`: inclusively defines out-of-sample periods, it is not meaningful for rolling method
* `--lags`: Time lags when making predictions. It will simple `shift` features forward. Normally set as 1, using `t-1` to predict `t`
* `--forward-bars`: If not given or given as 0, we are predicting daily returns (if `freq==d`), otherwise, a forward returns will be calculated as the labels. For example, `forward-bars==3` means returns from `t to t+2` and labeled at `t`

> Make sure you have all the data under `yahoos` folder, unzip the `data.zip` and put all csv files under `yahoos/data`.

After running the command, you should see code runnig on 600+ stocks in order. You can also specifiy certain tickers instead of all tickers by adding arguments `-t`. For example, `-t AAPL -t AMZN` will make the model run on two stocks only.

Then `predictions` and `metrics` will be stored as `csv` files under the same folder or you can specify you custom output path by specifying `--output` before `forecast` command.

> `metrics` will be t-statistics in the regression setup

#### Gradient Boosting

```
python3 main.py --name your_task_name \
                forecast \
                --model SKGBM \
                --config-path `pwd`/gbm.yaml \
                --minimum-train-bars 252 \
                --freq d \
                --train-periods 2001-01-01 2014-12-31 \
                --test-periods 2015-01-01 2018-12-31 \
                --lags 1 \
                --forward-bars 3
```
The current model for GBM is based on `sklearn.ensemble.GradientBoostingClassifier` and `scikit-optimize.BayesSearchCV`.

* `--model`: Use `SKGBM`
* `--config-path`: A config yaml file which specifies the hyper-parameters tuning configurations for the Bayesian Search CV. You can directly modify the default `gbm.yaml` file or create a new one based on the existing format.
* Any `rolling` related arguments can be remvoed, then this model will use the entire training periods to train, and make OOS predictions on test periods.

### How to customize new models

Check out the `models.py`. Any new model class should inherite from `ModelBase` class with foure must-implement functions. Then append this model into `ModelSelection` class.

For any models that follows `sklearn` api, simply create it under `ModelSelection`

```python
SKGBM = {
    "base": SklearnGeneralModel,  # Use Sklearn Template for any sklearn models
    "init_args": {
        "model": ens.GradientBoostingRegressor,  # The sklearn model class from skearln API
        "searchCV": True  # Set it True, if you want to hyper tune
    }
}
```

Then, in your `config-path` file, make sure all hyper-parameters are following this model's API.

### How to customize new data processors

Check out the `processor.py`. For any labels data (stock returns, etc.), follow `LabelProcessor`. For any features data (Fama-French, etc.) follow `FeatureProcessor`. Once created, append them into `TaskLabels` or `TaskFeatures` accordingly. Then in the terminal command, you can specify `--label` and `--features` arguments for those tasks.

The default settings are `--label yahoo` and `--features ff_basic`, which uses 600+ stocks' data from Yahoo and basic daily Fama-French factors.

## Calculated Results

As predictions and back-test files are normally large and repeated for different models, lags, forward-bars. They are stored seprately in a cloud storage instead of Github, please check here:
https://drive.google.com/open?id=1LE-HJXRAsuAgMgqL7iXY6S9rNIcSyqDQ

* **Predictions**: Predictions for each day's returns using different models
* **PerformanceMetrics**: `metrics.csv` recorded some specific per model metrics. In terms of Rolling Regression, this metric will be T-statstics. `BT.xlsx` recorded performance such as RMSE, MAE, R Square etc.
* **DataCache**: (!!Archived: As procesiing speed has been optimized, there is no need to store cache for the yahoo task!!)`pickle` file can be loaded directly into Python. Each file contains the original label values per stock. It can be N days forward returns.
