# tune-indicators
An attempt to find powerful combinations of technical indicators, through iterating variants with tuneta and backtesting

This program tests all indicators from pandas_ta, talib and finta, using tunata to find the "best" parameter settings. It then creates buy and sell signals based on indicator pairs and when one is larger than the other. It iterates through all possible combinations and saves the pair that generates best overall return for all stock tickers described in tickers.csv. By default, it fetches 5 years of stock data using Yahoo Finance.

Please note this program will run for a very long time if tickers.csv is a long list. Also, the multiprocessing can lead to errors in certain environments but it still likely needed to speed up the program.

