"""
pyperfanalytics.data
===================

Data loader utilities for packaged sample datasets.
"""

from importlib import resources
import pandas as pd


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the packaged data directory."""
    with resources.files("pyperfanalytics.data").joinpath(filename).open("r", encoding="utf-8") as f:
        return pd.read_csv(f, index_col=0, parse_dates=True)


def load_managers() -> pd.DataFrame:
    """
    Load the managers performance dataset.

    Contains monthly returns for several managers, asset classes, and the
    US 3-month Treasury Bill rate as a risk-free rate proxy.
    """
    return _load_csv("managers.csv")


def load_edhec() -> pd.DataFrame:
    """
    Load the EDHEC alternative investment strategy index returns dataset.

    Contains monthly returns for EDHEC hedge fund indices.
    """
    return _load_csv("edhec.csv")


def load_portfolio_bacon() -> pd.DataFrame:
    """
    Load the portfolio Bacon dataset.

    Contains monthly returns for a portfolio and its benchmark from Carl Bacon (2008).
    """
    return _load_csv("portfolio_bacon.csv")
