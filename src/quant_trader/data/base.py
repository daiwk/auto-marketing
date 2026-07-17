"""Market-data source boundary."""

from datetime import date
from typing import Protocol

import pandas as pd


class MarketDataSource(Protocol):
    """Fetch daily bars for [start, end): start is inclusive and end is exclusive."""

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...
