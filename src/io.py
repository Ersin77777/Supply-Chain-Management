from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_company_data(
    a_file: str | Path,
    b_file: str | Path,
    sep: str = ";",
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Company A and B datasets.
    Expected: CSV files (often ';' separated in Germany exports).
    """
    df_a = pd.read_csv(a_file, sep=sep, encoding=encoding)
    df_b = pd.read_csv(b_file, sep=sep, encoding=encoding)
    return df_a, df_b


def save_csv(df: pd.DataFrame, out_file: str | Path, index: bool = False) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=index)
