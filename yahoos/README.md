# S&P 500 FROM Yahoo

Pulled 639 / 950 tickers

Standard OLHCV tables pulled from Yahoo. The files can be found by unzipping `data.zip`.

## Extractors

Alternatively, one can download the data by using `python puller.py`, with all dependecies installed.

## Env Replicates

One can either

```
pip install -r requirements.txt
```

or 

```
conda env create -f environment.yml
```

## Failed Tickers

The tickers that are failed to pull from Yahoo are recorded at `log.txt`, we might need further investigations if we really need those data.
