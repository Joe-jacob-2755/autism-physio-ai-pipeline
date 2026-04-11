"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  encoder.py
=============================================================================
One-hot encoding for categorical demographic features.

After standardisation (RobustScaler), categorical demographics are
one-hot encoded for model ingestion. This produces separate columns
for each category value (e.g. gender_Male, gender_Female, gender_Non-binary).

Applied to both:
  - Combined feature CSV (all signals fused)
  - Individual signal feature CSVs

Both encoded and non-encoded versions are saved.

=============================================================================
"""
from __future__ import annotations

import pandas as pd
from typing import Dict

from config import DEMOGRAPHIC_CATEGORICAL_COLS, META_COLS


class OneHotEncoder:
    """
    One-hot encode categorical demographic columns in feature DataFrames.

    Keeps the original non-encoded DataFrame unchanged and produces a
    new DataFrame with categorical columns replaced by binary indicators.

    Parameters
    ----------
    categorical_cols : list of column names to one-hot encode
    drop_first       : drop first category to avoid multicollinearity
                       (default False — keep all for interpretability)
    verbose          : print encoding summary
    """

    def __init__(
        self,
        categorical_cols: list = None,
        drop_first: bool = False,
        verbose: bool = True,
    ):
        self.categorical_cols = categorical_cols or DEMOGRAPHIC_CATEGORICAL_COLS
        self.drop_first = drop_first
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical columns in a DataFrame.

        Parameters
        ----------
        df : feature DataFrame with categorical demographic columns

        Returns
        -------
        DataFrame with categorical columns replaced by one-hot indicators.
        Numeric features and meta columns are preserved unchanged.
        """
        df_out = df.copy()

        # Find which categorical columns are present
        cols_to_encode = [c for c in self.categorical_cols
                          if c in df_out.columns]

        if not cols_to_encode:
            self._log("[Encoder] No categorical columns found — "
                      "returning unchanged.")
            return df_out

        # Also drop the ordinal-encoded versions (e.g. gender_enc)
        enc_cols = [f"{c}_enc" for c in cols_to_encode
                    if f"{c}_enc" in df_out.columns]

        # One-hot encode
        df_encoded = pd.get_dummies(
            df_out,
            columns=cols_to_encode,
            drop_first=self.drop_first,
            dtype=int,
        )

        # Drop ordinal-encoded columns (replaced by one-hot)
        df_encoded = df_encoded.drop(columns=enc_cols, errors="ignore")

        n_new = df_encoded.shape[1] - df_out.shape[1] + len(cols_to_encode)
        self._log(f"[Encoder] One-hot encoded {len(cols_to_encode)} columns "
                  f"-> {n_new} new binary columns  "
                  f"(drop_first={self.drop_first})")

        return df_encoded

    def encode_all(
        self, feature_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        One-hot encode all signal feature DataFrames.

        Parameters
        ----------
        feature_dfs : {signal_name: DataFrame}

        Returns
        -------
        {signal_name: one-hot encoded DataFrame}
        """
        encoded = {}
        for sig_name, df in feature_dfs.items():
            encoded[sig_name] = self.encode_dataframe(df)
        return encoded

    def _log(self, msg):
        if self.verbose:
            print(msg)
