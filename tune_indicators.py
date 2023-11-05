import yfinance as yf
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from tuneta.tune_ta import TuneTA

class CustomException(Exception):
    def __init__(self, message="A custom exception occurred"):
        self.message = message
        super().__init__(self.message)

def clean_and_validate_data(stock_data):
    stock_data.dropna(inplace=True)
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data.dropna(inplace=True)

    close_column = next((col for col in stock_data.columns if col.lower() == 'close'), None)
    if close_column is None:
        raise CustomException("No 'Close' or 'close' column found in DataFrame")

    return stock_data, close_column

def download_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period="5y")
        if stock_data.empty:
            print(f"No data available for {ticker} for the last 5 years.")
            return None
        stock_data, _ = clean_and_validate_data(stock_data)
        return stock_data
    except Exception as e:
        raise CustomException(f"An error occurred while downloading {ticker}: {e}")

def find_best_indicators(stock_data, indicators=['all'], ranges=[(3, 150)], trials=500, early_stop=100, min_target_correlation=0.05):
    train_data = stock_data
    target_series_train = train_data['Close'].pct_change().shift(-1).dropna()

    tuneta = TuneTA(n_jobs=1, verbose=True)
    tuneta.fit(train_data.loc[target_series_train.index], target_series_train, indicators=indicators, ranges=ranges, trials=trials, early_stop=early_stop, min_target_correlation=min_target_correlation)
    tuneta.prune(max_inter_correlation=0.85)

    df_features_train = tuneta.transform(train_data)

    return df_features_train, tuneta

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def evaluate_indicator_pairs(self, stock_data, df_features, indicator1, indicator2):
        try:
            stock_data, close_column = clean_and_validate_data(stock_data)
        except CustomException as e:
            raise e

        transactions = []
        capital = self.initial_capital
        shares = 0
        num_trades = 0

        if not all(indicator in df_features.columns for indicator in [indicator1, indicator2]):
            return 0, 0, pd.DataFrame()

        buy_signals = df_features[indicator1] > df_features[indicator2]
        sell_signals = df_features[indicator1] < df_features[indicator2]

        for i, row in stock_data.iterrows():
            if i not in df_features.index:
                continue

            if shares > 0:
                capital = shares * row[close_column]

            if buy_signals.loc[i]:
                if shares == 0:
                    shares = capital / row[close_column]
                    capital = 0
                    transactions.append([i, 'BUY', row[close_column], shares, capital])
                    num_trades += 1

            elif sell_signals.loc[i]:
                if shares > 0:
                    capital = shares * row[close_column]
                    shares = 0
                    transactions.append([i, 'SELL', row[close_column], shares, capital])
                    num_trades += 1

        return capital, num_trades, pd.DataFrame(transactions, columns=['Date', 'Action', 'Price', 'Shares', 'Capital'])

def process_single_ticker(ticker, backtester):
    print(f"Processing ticker: {ticker}")

    stock_data = download_stock_data(ticker)
    if stock_data is None:
        return None, None
    df_features_train, tuneta = find_best_indicators(stock_data)

    performance_data = []
    indicators = df_features_train.columns

    for indicator1, indicator2 in combinations(indicators, 2):
        capital, trades, transactions = backtester.evaluate_indicator_pairs(stock_data, df_features_train, indicator1, indicator2)
        performance_data.append((indicator1, indicator2, capital, trades))

    performance_df = pd.DataFrame(performance_data, columns=['Indicator1', 'Indicator2', 'Capital', 'Trades'])

    return performance_df, tuneta

def process_ticker(ticker):
    try:
        backtester = Backtester()
        return process_single_ticker(ticker, backtester)
    except CustomException as e:
        print(f"Specific error processing {ticker}: {e}")
        return None, None
    except Exception as e:
        print(f"General error processing {ticker}: {e}")
        return None, None

def main():
    all_performance_data = []

    try:
        tickers = pd.read_csv('tickers.csv', header=None)[0].tolist()
    except FileNotFoundError:
        print("tickers.csv not found. Exiting.")
        return

    pool = mp.Pool(min(mp.cpu_count(), len(tickers)))
    results = pool.map(process_ticker, tickers)
    pool.close()
    pool.join()

    for result in results:
        if result is not None:
            performance_df, _ = result
            if performance_df is not None and not performance_df.empty:
                all_performance_data.append(performance_df)
            else:
                print(f"Performance data for ticker is empty or None.")

    if not all_performance_data:
        print("No valid performance data to concatenate. Exiting.")
        return

    global_performance_df = pd.concat(all_performance_data).groupby(['Indicator1', 'Indicator2']).mean()
    global_performance_df.to_csv('global_performance_metrics.csv')

    return global_performance_df

if __name__ == "__main__":
    main()
