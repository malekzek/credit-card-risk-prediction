from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BUREAU_BALANCE_FILE,
    BUREAU_FILE,
    CREDIT_CARD_BALANCE_FILE,
    INSTALLMENTS_FILE,
    POS_CASH_FILE,
    PREVIOUS_APPLICATION_FILE,
    PROCESSED_DATA_DIR,
    SECONDARY_FEATURES_FILE,
)


AGG_FUNCS = ("mean", "max", "min")
ID_COLUMNS = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU", "TARGET"}


def _read_csv_safely(csv_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV with encoding fallback to handle mixed-source files."""
    try:
        return pd.read_csv(csv_path, usecols=usecols, encoding="utf-8", encoding_errors="replace")
    except Exception:
        for enc in ("latin1", "cp1252"):
            try:
                return pd.read_csv(csv_path, usecols=usecols, encoding=enc)
            except Exception:
                continue
        raise


def _aggregate_numeric_by_curr(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate numeric columns to applicant level using SK_ID_CURR."""
    if "SK_ID_CURR" not in df.columns:
        raise ValueError(f"SK_ID_CURR is missing while aggregating {prefix}.")

    counts = df.groupby("SK_ID_CURR").size().rename(f"{prefix}_record_count")

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ID_COLUMNS]

    if not numeric_cols:
        return counts.reset_index()

    agg_df = df.groupby("SK_ID_CURR")[numeric_cols].agg(list(AGG_FUNCS))
    agg_df.columns = [
        f"{prefix}_{col}_{stat}".lower() for col, stat in agg_df.columns.to_flat_index()
    ]

    out = counts.to_frame().join(agg_df, how="left").reset_index()
    return out


def _build_bureau_features() -> pd.DataFrame:
    usecols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]
    bureau = _read_csv_safely(BUREAU_FILE, usecols=usecols)

    bureau["DEBT_CREDIT_RATIO"] = (
        bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    )
    return _aggregate_numeric_by_curr(bureau, prefix="bureau")


def _build_bureau_balance_features() -> pd.DataFrame:
    bureau_map = _read_csv_safely(BUREAU_FILE, usecols=["SK_ID_BUREAU", "SK_ID_CURR"])
    bureau_balance = _read_csv_safely(
        BUREAU_BALANCE_FILE,
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    )

    bureau_balance["STATUS_IS_DPD"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(
        "int8"
    )
    bureau_balance["STATUS_IS_UNKNOWN"] = bureau_balance["STATUS"].eq("X").astype("int8")

    bb_by_bureau = bureau_balance.groupby("SK_ID_BUREAU").agg(
        MONTHS_BALANCE_mean=("MONTHS_BALANCE", "mean"),
        MONTHS_BALANCE_max=("MONTHS_BALANCE", "max"),
        MONTHS_BALANCE_min=("MONTHS_BALANCE", "min"),
        STATUS_IS_DPD_mean=("STATUS_IS_DPD", "mean"),
        STATUS_IS_DPD_sum=("STATUS_IS_DPD", "sum"),
        STATUS_IS_UNKNOWN_mean=("STATUS_IS_UNKNOWN", "mean"),
    )
    bb_by_bureau = bb_by_bureau.reset_index()
    bb_by_bureau.columns = ["SK_ID_BUREAU"] + [
        f"bb_{col}".lower() for col in bb_by_bureau.columns[1:]
    ]

    merged = bureau_map.merge(bb_by_bureau, on="SK_ID_BUREAU", how="left")
    return _aggregate_numeric_by_curr(merged, prefix="bureau_balance")


def _build_previous_application_features() -> pd.DataFrame:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "HOUR_APPR_PROCESS_START",
        "NFLAG_LAST_APPL_IN_DAY",
        "RATE_DOWN_PAYMENT",
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "DAYS_DECISION",
        "SELLERPLACE_AREA",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
    ]
    prev = _read_csv_safely(PREVIOUS_APPLICATION_FILE, usecols=usecols)

    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)
    return _aggregate_numeric_by_curr(prev, prefix="previous")


def _build_pos_cash_features() -> pd.DataFrame:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    pos = _read_csv_safely(POS_CASH_FILE, usecols=usecols)
    return _aggregate_numeric_by_curr(pos, prefix="pos_cash")


def _build_installments_features() -> pd.DataFrame:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    installments = _read_csv_safely(INSTALLMENTS_FILE, usecols=usecols)

    installments["PAYMENT_DIFF"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    installments["PAYMENT_PERC"] = (
        installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"].replace(0, np.nan)
    )
    installments["DAYS_LATE"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)

    return _aggregate_numeric_by_curr(installments, prefix="installments")


def _build_credit_card_features() -> pd.DataFrame:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    cc = _read_csv_safely(CREDIT_CARD_BALANCE_FILE, usecols=usecols)

    cc["LIMIT_USE_RATIO"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["PAYMENT_MIN_RATIO"] = (
        cc["AMT_PAYMENT_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"].replace(0, np.nan)
    )

    return _aggregate_numeric_by_curr(cc, prefix="credit_card")


def build_secondary_features() -> pd.DataFrame:
    """Build applicant-level features from all secondary source tables."""
    print("Building bureau features...")
    bureau_features = _build_bureau_features()

    print("Building bureau balance features...")
    bureau_balance_features = _build_bureau_balance_features()

    print("Building previous application features...")
    previous_features = _build_previous_application_features()

    print("Building POS-CASH features...")
    pos_cash_features = _build_pos_cash_features()

    print("Building installments features...")
    installments_features = _build_installments_features()

    print("Building credit card features...")
    credit_card_features = _build_credit_card_features()

    frames = [
        bureau_features,
        bureau_balance_features,
        previous_features,
        pos_cash_features,
        installments_features,
        credit_card_features,
    ]

    all_features = frames[0]
    for frame in frames[1:]:
        all_features = all_features.merge(frame, on="SK_ID_CURR", how="outer")

    print(
        "Secondary features ready:",
        f"rows={all_features.shape[0]}, cols={all_features.shape[1]}",
    )
    return all_features


def load_or_build_secondary_features(force_rebuild: bool = False) -> pd.DataFrame:
    """Load secondary features cache or build it if cache is missing."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SECONDARY_FEATURES_FILE.exists() and not force_rebuild:
        print(f"Loading cached secondary features: {SECONDARY_FEATURES_FILE}")
        return pd.read_pickle(SECONDARY_FEATURES_FILE)

    features = build_secondary_features()
    features.to_pickle(SECONDARY_FEATURES_FILE)
    print(f"Saved secondary feature cache: {SECONDARY_FEATURES_FILE}")
    return features
