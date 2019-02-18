## AFP: Forecasting Scaffold

```
python3 main.py --name your_task_name 
                forecast \
                --model Regression \
                --rolling \
                --freq d \
                --train-periods 2001-01-01 2014-12-31 \
                --test-periods 2015-01-01 2018-12-31 \
                ----lags 20 \
                --forward-bars 20 \
                --label-cache `pwd`/yahoo_data_cache.pickle
```

More detailed documentations will be enhanced afterward...

Only need to expand the `processor` classes `model` classes. This scafold should work with more models and features/labels.

Next steps is to construct the a set of auto-analyzers on top of the generated predictions.

It would also be interesting to integrate a set of data transformation pipelines, but not as the priority. (As users can always "hardcode" those specific transformations into `model` or `processor` layers for ad-hoc purposes)
