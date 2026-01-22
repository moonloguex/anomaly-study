from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/agg/minute_metrics.parquet")
OUT = Path("data/agg/anomalies_feature_engineering.parquet")

# Rolling window size
WINDOWS = [5, 15, 60]

# Lag 간격
LAGS = [5, 10, 30]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # 시간 기반 특징 추가
    # 시간 (0-23)
    df["hour"] = df["ts"].dt.hour

    # 요일 (0=월, 6=일)
    df["day_of_week"] = df["ts"].dt.dayofweek

    # 주말 여부
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 업무 시간 (9시-18시)
    # 트래픽이 높고 에러율이 낮은게 정상
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype(int)

    # 새벽 시간 (1시-5시)
    # 트래픽이 낮고, 배치 작업으로 에러율 변동 가능성이 있음
    df["is_night"] = ((df["hour"] >= 1) & (df["hour"] < 5)).astype(int)

    return df

def add_rolling_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:

    for window in WINDOWS:
        # min_periods=1로 설정하여 초기값도 계산되도록 함
        
        # 이동 평균, 노이즈를 제거하고 트렌드를 보여줌
        df[f"{column}_rolling_mean_{window}m"] = df.groupby("service")[column].transform(lambda x: x.rolling(window, min_periods=1).mean())

        # 이동 표준편차, 변동성을 나타냄, 크면 불안정
        df[f"{column}_rolling_std_{window}m"] = df.groupby("service")[column].transform(lambda x: x.rolling(window, min_periods=1).std())

        # 이동 최솟값
        df[f"{column}_rolling_min_{window}m"] = df.groupby("service")[column].transform(lambda x: x.rolling(window, min_periods=1).min())

        # 이동 최댓값
        df[f"{column}_rolling_max_{window}m"] = df.groupby("service")[column].transform(lambda x: x.rolling(window, min_periods=1).max())

        # 현재 값과 평균값의 비교
        # (현재값 - 평균값) / 표준편차 = Z-score 개념
        mean_col = f"{column}_rolling_mean_{window}m"
        std_col = f"{column}_rolling_std_{window}m"

        df[f"{column}_zscore_{window}m"] = (df[column] - df[mean_col]) / (df[std_col] + 1e-9)

    return df

def add_lag_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    
    for lag in LAGS:
        # shift 함수로 이전 시점 값 가져오기
        # lag 5면 5분 전 값
        df[f"{column}_lag_{lag}m"] = df.groupby("service")[column].shift(lag)

        # 변화량 = 현재 - 과거
        df[f"{column}_diff_{lag}m"] = df[column] - df[f"{column}_lag_{lag}m"]

        # 변화 비율 = (현재 - 과거) / 과거
        df[f"{column}_pct_change_{lag}m"] = df[f"{column}_diff_{lag}m"] / (df[f"{column}_lag_{lag}m"] + 1e-9)

    return df

def main() -> None:

    # 데이터 로드
    df = pd.read_parquet(DATA)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["service", "ts"]).reset_index(drop=True)

    # 시간 특성 추가
    df = add_time_features(df)

    # 통계 특성 추가
    df = add_rolling_stats(df, "error_rate")

    # Lag 특성 추가
    df = add_lag_features(df, "error_rate")

    # 추가 특성 계산
    # 에러 밀도 = 에러수 / 요청수
    df["error_density"] = df["err_count"] / df["req_count"].clip(lower=1)

    # 로그 변환
    df["log_req_count"] = np.log1p(df["req_count"])
    df["log_err_count"] = np.log1p(df["err_count"])

    # 요청 대비 에러 비율
    df["err_req_ratio"] = df["err_count"] / df["req_count"].clip(lower=1)

    # 결측치 처리
    # Lag 특성 등에서 생긴 결측치를 0으로 채움
    nan_count_before = df.isnull().sum().sum()

    # 숫자 컬럼만 선택해서 0으로 채우기
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    nan_count_after = df.isnull().sum().sum()

    # 저장
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # 샘플 확인
    sample_service = df["service"].iloc[0]
    sample = df[df["service"] == sample_service].tail(10)

    display_cols = [
        "ts", "service", "error_rate",
        "hour", "is_weekend", "is_business_hours",
        "error_rate_rolling_mean_60m",
        "error_rate_zscore_60m",
        "error_rate_lag_5m",
        "error_rate_pct_change_5m"
    ]

    print(sample[display_cols].to_string(index=False))

    # 특성 중요도
    # 변동성이 큰 특성 찾기
    feature_cols = [col for col in df.columns if col not in ["ts", "service", "req_count", "err_count", "error_rate", "p95_latency_ms"]]

    feature_stats = df[feature_cols].std().sort_values(ascending=False)

    print(feature_stats.head(10).to_string())

if __name__ == "__main__":
    main()