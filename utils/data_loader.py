import yfinance as yf
import numpy as np
import pandas as pd

class RealDataLoader:
    """
    Handles downloading and preprocessing of real financial data for SBTS.
    """
    def __init__(self, ticker, start_date, end_date, interval='1d'):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.interval = interval
        self.raw_data = None
        self.returns = None

    def download(self):
        print(f"   [Data] Downloading {self.ticker} from {self.start} to {self.end}...")
        self.raw_data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval, progress=False)
        
        if len(self.raw_data) == 0:
            raise ValueError(f"No data found for {self.ticker}. Check internet or ticker symbol.")
            
        # Keep only Close price (or Adj Close)
        # Handle multi-index columns if they exist (yfinance update behavior)
        if 'Adj Close' in self.raw_data.columns:
            self.prices = self.raw_data['Adj Close']
        elif 'Close' in self.raw_data.columns:
            self.prices = self.raw_data['Close']
        else:
            self.prices = self.raw_data.iloc[:, 0]
            
        print(f"   [Data] Downloaded {len(self.prices)} data points.")
        return self

    def preprocess_to_returns(self):
        """
        Converts Prices to Log Returns.
        r_t = ln(S_t / S_{t-1})
        """
        if self.prices is None:
            self.download()
            
        # Log returns are better for SDEs than simple percentage returns
        # They are time-additive and usually more normally distributed
        prices_arr = self.prices.values.flatten()
        
        # Avoid division by zero or logs of zero
        prices_arr = prices_arr[prices_arr > 0]
        
        self.log_returns = np.log(prices_arr[1:] / prices_arr[:-1])
        
        # Remove outliers (Optional but recommended for stability)
        # e.g., trim returns > 5 std devs
        mean = np.mean(self.log_returns)
        std = np.std(self.log_returns)
        self.log_returns = np.clip(self.log_returns, mean - 5*std, mean + 5*std)
        
        return self.log_returns

    def get_sliding_windows(self, seq_len):
        """
        Creates (N, seq_len, 1) dataset using sliding windows.
        """
        if not hasattr(self, 'log_returns'):
            self.preprocess_to_returns()
            
        data = self.log_returns
        num_windows = len(data) - seq_len + 1
        
        if num_windows <= 0:
            raise ValueError(f"Sequence length {seq_len} is too long for data length {len(data)}")
            
        windows = []
        for i in range(num_windows):
            window = data[i : i + seq_len]
            windows.append(window)
            
        # Shape: (N_samples, seq_len, 1)
        dataset = np.array(windows)[..., np.newaxis]
        
        print(f"   [Data] Created {len(dataset)} samples of length {seq_len} from real data.")
        return dataset, np.mean(data), np.std(data)

def reconstruct_prices(initial_price, log_returns_paths):
    """
    Helper to convert generated Log Returns back to Price Paths.
    S_t = S_0 * exp(cumsum(r))
    """
    # log_returns_paths: (N_paths, T, 1)
    # cumsum along time axis
    cum_returns = np.cumsum(log_returns_paths, axis=1)
    
    # Add initial 0 to cumsum to include S0
    # But usually we generate T steps forward.
    
    # Shape matching broadcasting
    price_paths = initial_price * np.exp(cum_returns)
    
    # Prepend initial price
    N, T, D = price_paths.shape
    start_block = np.full((N, 1, D), initial_price)
    full_paths = np.concatenate([start_block, price_paths], axis=1)
    
    return full_paths