"""Offline tests for psx-data-reader.

No network: parsing/coercion is exercised directly against a fixed PSX-like
HTML fixture, and the network boundary (``download``) is monkeypatched so the
``stocks``/``get_psx_data`` orchestration can be tested without hitting
``dps.psx.com.pk``.
"""

import datetime
import logging

import pandas as pd
from bs4 import BeautifulSoup

from psx.web import DataReader

# A representative slice of the PSX historical endpoint: a <th> header row,
# two normal rows, and a no-trades row (VOLUME == "-").
SAMPLE_HTML = """
<table>
<tr><th>DATE</th><th>OPEN</th><th>HIGH</th><th>LOW</th><th>CLOSE</th><th>VOLUME</th></tr>
<tr><td>Jan 02, 2022</td><td>100.00</td><td>101.00</td><td>99.00</td><td>100.50</td><td>1,200</td></tr>
<tr><td>Jan 03, 2022</td><td>100.50</td><td>102.00</td><td>100.00</td><td>101.50</td><td>-</td></tr>
<tr><td>Jan 05, 2022</td><td>101.50</td><td>103.00</td><td>101.00</td><td>102.50</td><td>2,500</td></tr>
</table>
"""

# Adds a short row (4 <td>) and an empty-date row — both must be dropped.
MALFORMED_HTML = """
<table>
<tr><th>DATE</th><th>OPEN</th><th>HIGH</th><th>LOW</th><th>CLOSE</th><th>VOLUME</th></tr>
<tr><td>Jan 02, 2022</td><td>100.00</td><td>101.00</td><td>99.00</td><td>100.50</td><td>1,200</td></tr>
<tr><td>Jan 03, 2022</td><td>100.50</td><td>102.00</td><td>100.00</td><td>101.50</td><td>-</td></tr>
<tr><td>Jan 05, 2022</td><td>101.50</td><td>103.00</td><td>101.00</td><td>102.50</td><td>2,500</td></tr>
<tr><td>Jan 06, 2022</td><td>102.00</td><td>103.00</td><td>102.00</td></tr>
<tr><td></td><td>x</td><td>y</td><td>z</td><td>w</td><td>v</td></tr>
</table>
"""


def _soup(html):
    return BeautifulSoup(html, "html.parser")


def test_toframe_parses_dates_and_preserves_raw_cells():
    dr = DataReader()
    df = dr.toframe(_soup(SAMPLE_HTML))
    assert list(df.columns) == ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    assert list(df.index) == [
        datetime.datetime(2022, 1, 2),
        datetime.datetime(2022, 1, 3),
        datetime.datetime(2022, 1, 5),
    ]
    # toframe only parses the DATE column; numerics stay as raw strings
    # (commas / "-") for preprocess to clean.
    assert df.loc[datetime.datetime(2022, 1, 2), "VOLUME"] == "1,200"
    assert df.loc[datetime.datetime(2022, 1, 3), "VOLUME"] == "-"


def test_toframe_skips_malformed_and_unparseable_rows(caplog):
    dr = DataReader()
    with caplog.at_level(logging.WARNING):
        df = dr.toframe(_soup(MALFORMED_HTML))
    # 3 well-formed rows survive; the short row and the empty-date row drop.
    assert len(df) == 3
    # The <th> header row is expected and must NOT count as dropped —
    # only the 2 genuinely malformed data rows do.
    assert any("Skipped 2" in r.message for r in caplog.records)


def test_toframe_clean_response_emits_no_warning(caplog):
    dr = DataReader()
    with caplog.at_level(logging.WARNING):
        dr.toframe(_soup(SAMPLE_HTML))
    assert not [r for r in caplog.records if "Skipped" in r.message]


def test_toframe_empty_response_returns_empty_frame():
    dr = DataReader()
    df = dr.toframe(_soup("<table></table>"))
    assert df.empty


def test_preprocess_coerces_and_handles_dash():
    dr = DataReader()
    processed = dr.preprocess([dr.toframe(_soup(SAMPLE_HTML))])
    assert list(processed.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert processed.index.name == "Date"
    # commas stripped to float
    assert processed.loc[datetime.datetime(2022, 1, 2), "Volume"] == 1200.0
    # no-trades "-" becomes NaN, not a crash
    assert pd.isna(processed.loc[datetime.datetime(2022, 1, 3), "Volume"])
    # every column is float
    assert all(pd.api.types.is_float_dtype(processed[c]) for c in processed.columns)


def test_get_psx_data_preprocesses_mocked_download(monkeypatch):
    dr = DataReader()
    raw = dr.toframe(_soup(SAMPLE_HTML))
    monkeypatch.setattr(dr, "download", lambda symbol: raw)
    out = dr.get_psx_data("DUMMY")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.loc[datetime.datetime(2022, 1, 2), "Close"] == 100.5


def test_stocks_single_ticker_slices(monkeypatch):
    dr = DataReader()
    processed = dr.preprocess([dr.toframe(_soup(SAMPLE_HTML))])
    monkeypatch.setattr(dr, "get_psx_data", lambda ticker: processed)
    # label slice on a DatetimeIndex is inclusive of both endpoints
    out = dr.stocks("DUMMY", datetime.date(2022, 1, 3), datetime.date(2022, 1, 5))
    assert datetime.datetime(2022, 1, 2) not in out.index
    assert datetime.datetime(2022, 1, 3) in out.index
    assert datetime.datetime(2022, 1, 5) in out.index


def test_stocks_multiple_tickers_concat_with_keys(monkeypatch):
    dr = DataReader()
    processed = dr.preprocess([dr.toframe(_soup(SAMPLE_HTML))])
    monkeypatch.setattr(dr, "get_psx_data", lambda ticker: processed)
    out = dr.stocks(["A", "B"], datetime.date(2022, 1, 1), datetime.date(2022, 1, 31))
    # MultiIndex keyed by ticker, then date
    assert out.index.names == ["Ticker", "Date"]
    assert "A" in out.index.get_level_values("Ticker")
    assert "B" in out.index.get_level_values("Ticker")
