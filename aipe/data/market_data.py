import ccxt
import pandas as pd
from rich.console import Console

console = Console()

def get_market_data(ticker="XRP/USDT", timeframe='1d', limit=500):
    """
    Downloads historical market data from Binance.

    Args:
        ticker (str): The stock ticker.
        timeframe (str): The timeframe to download data for.
        limit (int): The number of data points to download.

    Returns:
        A pandas DataFrame with the historical data.
    """
    console.log(f"Downloading market data for {ticker} (timeframe: {timeframe}, limit: {limit})...")

    exchange = ccxt.kucoin()

    try:
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        console.log("Market data download complete.")
        return df
    except Exception as e:
        console.log(f"[bold red]Error downloading market data: {e}[/bold red]")
        return None
