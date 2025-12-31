from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/agg/minute_metrics.parquet")

# 튜닝 파라미터 민감도 레버
WINDOW = 180       # p(정상 에러율)를 추정할 때 과거 몇 분을 볼지(180분 = 3시간)
Z_TH = 6.0         # binomial z-score 임계값(높일수록 덜 민감)
MIN_REQ = 80       # 요청 수가 너무 적으면 통계적으로 불안정하므로 제외
MIN_ERR = 3        # 에러 1~2건은 흔한 노이즈일 수 있어 최소 에러 건수 조건
CONSEC_N = 3       # 연속 N회 이상이면 알람(스파이크 1회성 오탐 완화)
EPS = 1e-12        # 0으로 나누는 문제를 피하기 위한 아주 작은 수

def rolling_p_hat(err: pd.Series, req: pd.Series) -> pd.Series:
    # 과거 WINDOW 분 동안의 누적 에러율 추정치 계산
    err_sum = df.groupby("service")["err_count"].transform(
        lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum()
    )
    req_sum = df.groupby("service")["req_count"].transform(
        lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum()
    )
    return (err_sum / req_sum.clip(lower=1)).clip(lower=1e-6, upper=1 - 1e-6)

def main() -> None:
    df = pd.read_parquet(DATA)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["service", "ts"])

    df["err_count"] = df["err_count"].clip(lower=0)
    df["req_count"] = df["req_count"].clip(lower=0)

    # 정상 에러율 추정 p_hat = (과거 에러 총합) / (과거 요청 총합)
    df["p_hat"] = (err_sum / req_sum.clip(lower=1)).clip(lower=1e-6, upper=1 - 1e-6)

    # 기대 에러 건수 = n * p
    df["expected_err"] = df["req_count"] * df["p_hat"]
    # 분산 = n * p * (1 - p)  (Binomial의 분산)
    df["var_err"] = df["req_count"] * df["p_hat"] * (1 - df["p_hat"])
    # 표준편차 = sqrt(분산)
    df["std_err"] = np.sqrt(df["var_err"] + EPS)
    # z = (관측치 - 기대치) / 표준편차
    df["z_binom"] = (df["err_count"] - df["expected_err"]) / df["std_err"]

    df["is_candidate"] = (
        (df["z_binom"] > Z_TH) &
        (df["req_count"] >= MIN_REQ) &
        (df["err_count"] >= MIN_ERR)
    )

    # 연속 N회 이상인 경우만 이상으로 판정
    df["candidate_run"] = (
        df.groupby("service")["is_candidate"]
        .transform(lambda s: s.rolling(CONSEC_N, min_periods=CONSEC_N).sum())
    )

    df["is_anomaly"] = df["candidate_run"] >= CONSEC_N

    out = df[df["is_anomaly"]].copy().sort_values(["service", "ts"])

    print("\n[summary] anomaly counts by service:")
    if len(out) == 0:
        print("No anomalies detected.")
    else:
        print(out["service"].value_counts(dropna=False).to_string())

    print("\n[anomalies] (top 50):")
    cols = ["ts", "service", "req_count", "err_count", "error_rate", "p_hat", "z_binom", "p95_latency_ms"]
    print(out[cols].head(50).to_string(index=False))

if __name__ == "__main__":
    main()