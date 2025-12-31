from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/agg/minute_metrics.parquet")

# 과거 60분 기반, robust z 임계값(초기값), 저트래픽 오탐 방지 최소 요청수
#WINDOW = 60
#Z_TH = 3.5
#MIN_REQ = 50
#EPS = 1e-9

# 탐지 민감도 조절용 파라미터
WINDOW = 60
Z_TH = 4.5
MIN_REQ = 50
MIN_ERR = 2
MAD_FLOOR = 1e-4
EPS = 1e-12

def robust_z(series: pd.Series) -> pd.Series:
    # 중간값과 중앙값 절대 편차(MAD) 기반으로 robust z 점수 계산
    # 0.6745는 정규분포 가정시 MAD와 표준편차의 관계에서 유도된 상수
    med = series.rolling(WINDOW, min_periods=WINDOW).median()
    mad = (series - med).abs().rolling(WINDOW, min_periods=WINDOW).median()
    # MAD가 너무 작을 경우, 정확한 z 점수 계산을 위해 최소값으로 클리핑
    mad = mad.clip(lower=MAD_FLOOR)
    return 0.6745 * (series - med) / (mad + EPS)

def main() -> None:
    df = pd.read_parquet(DATA)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["service", "ts"])

    #df["z_error_rate"] = df.groupby("service")["error_rate"].transform(robust_z)
    df["smoothed_error_rate"] = (df["err_count"] + 1) / (df["req_count"].clip(lower=1) + 2)

    #df["is_anomaly"] = (df["z_error_rate"].abs() > Z_TH) & (df["req_count"] >= MIN_REQ)
    df["z_smoothed_error_rate"] = df.groupby("service")["smoothed_error_rate"].transform(robust_z)

    # 이상 판정
    df["is_anomaly"] = (
        (df["z_smoothed_error_rate"].abs() > Z_TH) &
        (df["req_count"] >= MIN_REQ) &
        (df["err_count"] >= MIN_ERR)
    )

    out = df[df["is_anomaly"]].copy()
    out = out.sort_values(["service", "ts"])

    print("\n[summary] anomaly counts by service:")
    print(out["service"].value_counts(dropna=False).to_string())

    print("\n[anomalies] (top 50):")
    cols = ["ts", "service", "req_count", "err_count", "error_rate", "smoothed_error_rate", "z_smoothed_error_rate", "p95_latency_ms"]
    print(out[cols].head(50).to_string(index=False))

if __name__ == "__main__":
    main()