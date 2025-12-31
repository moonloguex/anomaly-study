import pandas as pd

df = pd.read_parquet("data/agg/minute_metrics.parquet")

# api-gateway 서비스 새벽 구간 (error_rate) 분포 0 확인
g = df[df["service"] == "api-gateway"].copy()
g["ts"] = pd.to_datetime(g["ts"])
night = g[(g["ts"].dt.hour >= 1) & (g["ts"].dt.hour <= 5)]

print("night rows:", len(night))
print("error_rate == 0 ratio:", (night["error_rate"] == 0).mean())
print("req_count min/median:", int(night["req_count"].min()), " / ", int(night["req_count"].median()))