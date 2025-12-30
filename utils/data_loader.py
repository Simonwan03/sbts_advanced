import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import time

class RealDataLoader:
    """
    Handles downloading, aligning, and preprocessing real financial data 
    for SBTS, supporting multiple assets (Scale: 1 to 500+).
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str, interval: str = '1d'):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.interval = interval
        self.raw_prices_df = None 
        self.log_returns = None   

    @staticmethod
    def get_sp500_tickers(limit=None):
        """
        Scrapes S&P 500 tickers from Wikipedia.
        """
        print("   [Data] Fetching S&P 500 tickers from Wikipedia...")
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            tickers = df['Symbol'].tolist()
            # Fix some tickers (e.g., BRK.B -> BRK-B)
            tickers = [t.replace('.', '-') for t in tickers]
            if limit:
                return tickers[:limit]
            return tickers
        except Exception as e:
            print(f"   [Error] Failed to fetch S&P 500 tickers: {e}")
            return ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'] # Fallback

    def download(self):
        """
        Downloads data for all tickers using yfinance batch download.
        Handles dropouts and alignment automatically.
        """
        n_tickers = len(self.tickers)
        print(f"   [Data] Downloading {n_tickers} assets from {self.start} to {self.end}...")
        
        # Use yfinance batch download which is much faster for many stocks
        # auto_adjust=True handles splits/dividends
        try:
            df = yf.download(self.tickers, start=self.start, end=self.end, interval=self.interval, 
                             progress=True, auto_adjust=True)
        except Exception as e:
            print(f"   [Data] Batch download failed, retrying without auto_adjust: {e}")
            df = yf.download(self.tickers, start=self.start, end=self.end, interval=self.interval, 
                             progress=True)

        # Extract Close or Adj Close
        if 'Close' in df.columns and isinstance(df.columns, pd.MultiIndex):
            # yfinance returns MultiIndex (Price, Ticker)
            prices = df['Close']
        elif 'Adj Close' in df.columns:
            prices = df['Adj Close']
        else:
            # Fallback if structure is flat (single ticker)
            prices = df

        # --- Data Cleaning ---
        # 1. Drop columns (stocks) that are completely empty or have too many NaNs
        prices = prices.dropna(axis=1, how='all')
        
        # 2. Drop rows (dates) with any missing data (Strict alignment)
        # For 500 stocks, dropping rows with ANY NaN might kill all data.
        # Strategy: Drop stocks with > 5% missing data, then drop rows.
        missing_frac = prices.isnull().mean()
        valid_tickers = missing_frac[missing_frac < 0.05].index
        prices = prices[valid_tickers]
        
        initial_rows = len(prices)
        prices = prices.dropna()
        
        print(f"   [Data] Downloaded. Retained {prices.shape[1]}/{n_tickers} assets and {len(prices)}/{initial_rows} time steps.")
        
        self.raw_prices_df = prices
        return self

    def preprocess_to_returns(self):
        if self.raw_prices_df is None:
            self.download()
            
        # Log Returns
        # epsilon to avoid log(0) if data has 0s (unlikely for Close price but safe)
        prices_val = self.raw_prices_df.values
        log_returns_df = np.log(prices_val[1:] / (prices_val[:-1] + 1e-8))
        
        # Robust Clipping (Global or Per-Asset)
        # Per-Asset is better for diverse portfolio
        mean_ret = np.nanmean(log_returns_df, axis=0)
        std_ret = np.nanstd(log_returns_df, axis=0)
        
        # Broadcasting clipping
        lower = mean_ret - 5 * std_ret
        upper = mean_ret + 5 * std_ret
        
        self.log_returns = np.clip(log_returns_df, lower, upper)
        self.log_returns = np.nan_to_num(self.log_returns)
        return self.log_returns

    def get_sliding_windows(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, 'log_returns') or self.log_returns is None:
            self.preprocess_to_returns()
            
        data = self.log_returns
        num_windows = data.shape[0] - seq_len + 1
        
        if num_windows <= 0:
            raise ValueError(f"Sequence length {seq_len} too long.")
            
        windows = []
        for i in range(num_windows):
            window = data[i : i + seq_len]
            windows.append(window)
            
        dataset = np.array(windows) # (N, T, D)
        
        # Ensure 3D
        if dataset.ndim == 2:
            dataset = dataset[..., np.newaxis]

        mean_ret = np.mean(data, axis=0)
        std_ret = np.std(data, axis=0)
        
        print(f"   [Data] Tensor Ready. Shape: {dataset.shape}")
        return dataset, mean_ret, std_ret

def reconstruct_prices(initial_prices: np.ndarray, log_returns_paths: np.ndarray) -> np.ndarray:
    """
    Reconstructs prices for multi-asset paths.
    """
    log_returns_paths = np.asarray(log_returns_paths)
    initial_prices = np.asarray(initial_prices)
    
    if log_returns_paths.ndim == 2: 
        log_returns_paths = log_returns_paths[..., np.newaxis]
    
    N_paths, T_steps, N_assets = log_returns_paths.shape
    
    # 1. Scalar Initial Price -> Broadcast to (N_paths,)
    if initial_prices.ndim == 0:
        initial_prices = np.full(N_paths, initial_prices)
    
    # 2. Cumulative Sum
    cum_returns = np.cumsum(log_returns_paths, axis=1) 
    
    # 3. Reshape S0 for (N_paths, 1, 1) to broadcast over Time and Assets?
    # NO. S0 usually is scalar 100 for all assets, OR a vector of size N_Assets.
    # If S0 is scalar, we assume normalized starting price for ALL assets.
    
    # Let's assume initial_prices is (N_paths,) and applies to ALL assets (normalized view)
    S0_reshaped = initial_prices[:, np.newaxis, np.newaxis]
    
    price_paths = S0_reshaped * np.exp(cum_returns)
    
    # Add t=0
    start_block = S0_reshaped.repeat(N_assets, axis=2) # (N, 1, D)
    full_paths = np.concatenate([start_block, price_paths], axis=1)
    
    return full_paths