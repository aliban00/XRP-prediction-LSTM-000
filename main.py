from aipe.data.market_data import get_market_data
from aipe.interface.orchestrator import Orchestrator

if __name__ == "__main__":
    # Download real market data for XRP/USDT
    data = get_market_data(ticker="XRP/USDT", timeframe='1d', limit=500)

    if data is not None:
        # We'll use multiple features for our analysis
        data_for_aipe = data[['open', 'high', 'low', 'close', 'volume']]

        orchestrator = Orchestrator(data_for_aipe)
        orchestrator.run(data_for_aipe)
