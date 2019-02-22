## AFP: Forecasting Scaffold

```
python3 main.py --name your_task_name \
                --output $(dirname `pwd`)/forecasts_outputs \
                forecast \
                --model Regression \
                --rolling \
                --rolling-bars 252 \
                --predict-bars 20 \
                --minimum-train-bars 252 \
                --freq d \
                --train-periods 2001-01-01 2014-12-31 \
                --test-periods 2015-01-01 2018-12-31 \
                --lags 20 \
                --forward-bars 20 \
                --label-cache `pwd`/yahoo_data_cache_20d.pickle \
                --multiprocess
```

More detailed documentations will be enhanced afterward...

Only need to expand the `processor` classes `model` classes. This scafold should work with more models and features/labels.

Next steps is to construct the a set of auto-analyzers on top of the generated predictions.

It would also be interesting to integrate a set of data transformation pipelines, but not as the priority. (As users can always "hardcode" those specific transformations into `model` or `processor` layers for ad-hoc purposes)

> The cleaned 20 days forward rate Yahoo data cache can also be downloaded here: https://www.dropbox.com/s/zoczdur4m2gxbuk/yahoo_data_cache.pickle?dl=0

## Calculated Results

Some results can be found here:
https://drive.google.com/open?id=1LE-HJXRAsuAgMgqL7iXY6S9rNIcSyqDQ

* **Predictions**: Predictions for each day's returns using different models
* **PerformanceMetrics**: `metrics.csv` recorded some specific per model metrics. In terms of Rolling Regression, this metric will be T-statstics. `BT.xlsx` recorded performance such as RMSE, MAE, R Square etc.
* **DataCache**: `pickle` file can be loaded directly into Python. Each file contains the original label values per stock. It can be N days forward returns.
