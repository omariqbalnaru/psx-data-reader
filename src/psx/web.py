import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Union

import pandas as pd
import requests
from bs4 import BeautifulSoup as parser
from pandas import DataFrame as container
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataReader:
    headers = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

    def __init__(self):
        self.__history = "https://dps.psx.com.pk/historical"
        self.__symbols = "https://dps.psx.com.pk/symbols"
        self.__local = threading.local()

    @property
    def session(self):
        if not hasattr(self.__local, "session"):
            self.__local.session = requests.Session()
        return self.__local.session

    def tickers(self):
        return pd.read_json(self.__symbols)

    def get_psx_data(self, symbol: str) -> container:
        """Fetch and clean the complete history for a single symbol.

        The PSX historical endpoint returns every available row for a symbol in
        a single request (month=0, year=0), so one download is enough. Any date
        windowing is applied locally afterwards by ``stocks``.
        """
        data = self.download(symbol)
        if not isinstance(data, container) or data.empty:
            return data
        return self.preprocess([data])

    def stocks(self, tickers: Union[str, list], start: date, end: date) -> container:
        tickers = [tickers] if isinstance(tickers, str) else tickers

        # One request per symbol returns its full history, so fetch symbols in
        # parallel and slice to [start, end] locally instead of re-downloading
        # the same full history once per month.
        with tqdm(total=len(tickers), desc="Downloading data") as progressbar:
            with ThreadPoolExecutor(max_workers=max(1, min(6, len(tickers)))) as executor:
                future_to_ticker = {
                    executor.submit(self.get_psx_data, ticker): ticker for ticker in tickers
                }
                data = []
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    frame = future.result()
                    progressbar.update(1)
                    if isinstance(frame, container):
                        data.append((ticker, frame))

        if len(data) == 1:
            return data[0][1][start:end]

        frames = [frame[start:end] for _, frame in data]
        keys = [ticker for ticker, _ in data]
        return pd.concat(frames, keys=keys, names=["Ticker", "Date"])

    def download(self, symbol: str):
        session = self.session
        # month=0, year=0 asks the PSX API for the symbol's complete history.
        post = {"month": "0", "year": "0", "symbol": symbol}
        with session.post(self.__history, data=post, timeout=30) as response:
            parsed = parser(response.text, features="html.parser")
        return self.toframe(parsed)

    def toframe(self, data):
        stocks = defaultdict(list)
        rows = data.select("tr")
        dropped = 0

        for row in rows:
            cols = [col.getText() for col in row.select("td")]
            # Header rows use <th> (no <td>) — expected, not a data-quality
            # issue, so skip silently. Partial data rows would misalign
            # columns if zipped as-is, so drop and count those.
            if len(cols) != len(self.headers):
                if cols:
                    dropped += 1
                continue

            try:
                cols[0] = datetime.strptime(cols[0], "%b %d, %Y")
            except ValueError:
                # Unparseable date (e.g. empty cell) — drop the whole row.
                dropped += 1
                continue

            for key, value in zip(self.headers, cols):
                stocks[key].append(value)

        if dropped:
            logger.warning("Skipped %d malformed/unparseable row(s) from PSX response", dropped)

        return pd.DataFrame(stocks, columns=self.headers).set_index("DATE")

    def preprocess(self, data: list) -> pd.DataFrame:
        # concatenate each frame to a single dataframe
        data = pd.concat(data)
        # sort the data by date
        data = data.sort_index()
        # change indexes from all uppercase to title
        data = data.rename(columns=str.title)
        # change index label Title to Date
        data.index.name = "Date"
        # coerce each column type to float; non-numeric cells (e.g. "-" on a
        # no-trades day) become NaN instead of raising.
        for column in data.columns:
            cleaned = data[column].astype(str).str.replace(",", "", regex=False)
            data[column] = pd.to_numeric(cleaned, errors="coerce")
        return data


data_reader = DataReader()

if __name__ == "__main__":
    data = data_reader.stocks("OGDC", date(2018, 12, 10), date(2019, 12, 10))
