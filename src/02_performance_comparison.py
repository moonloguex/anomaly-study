import time
import pandas as pd
import duckdb
from pathlib import Path

RAW = Path("data/raw/events.jsonl")
OUT_PANDAS = Path("data/agg/minute_metrics_pandas.parquet")
OUT_DUCKDB = Path("data/agg/minute_metrics_duckdb.parquet")

# Pandas로 처리
start = time.time()

df = pd.read_json(RAW, lines=True)

df["ts"] = pd.to_datetime(df["ts"])
df["error_rate"] = df.apply(lambda row: row["err_count"] / row["req_count"] if row["req_count"] > 0 else 0.0, axis=1)

# Parquet로 저장
df.to_parquet(OUT_PANDAS, index=False)

pandas_time = time.time() - start
print(f"Pandas processing time: {pandas_time:.4f} seconds")
# Pandas processing time: 0.5865 seconds
# Pandas processing time: 0.5183 seconds
# Pandas processing time: 0.5117 seconds

# DuckDB로 처리
start = time.time()

con = duckdb.connect(database=":memory:")
con.execute("INSTALL json; LOAD json;")

raw = str(RAW.as_posix()).replace("'", "''")
out = str(OUT_DUCKDB.as_posix()).replace("'", "''")

con.execute(f"""
    COPY (
        SELECT CAST(ts AS TIMESTAMP) AS ts
             , service
             , req_count
             , err_count
             , CASE WHEN req_count > 0 THEN err_count::DOUBLE / req_count ELSE 0.0 END AS error_rate
             , p95_latency_ms
          FROM read_json_auto('{raw}')
    ) TO '{out}' (FORMAT PARQUET);
""")

duckdb_time = time.time() - start
print(f"DuckDB processing time: {duckdb_time:.4f} seconds")
# DuckDB processing time: 0.0399 seconds
# DuckDB processing time: 0.0371 seconds
# DuckDB processing time: 0.0366 seconds

con.close()