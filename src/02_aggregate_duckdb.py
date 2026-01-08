from __future__ import annotations

from pathlib import Path
import duckdb

RAW = Path("data/raw/events.jsonl")
OUT = Path("data/agg/minute_metrics.parquet")

def sql_str_path(p: Path) -> str:
    # SQL 문자열 리터럴로 안전하게 넣기 위해 ' 이스케이프
    return str(p.as_posix()).replace("'", "''")

def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL json;")
    con.execute("LOAD json;")

    raw = sql_str_path(RAW.resolve())
    out = sql_str_path(OUT.resolve())

    # JSONL 로드 - 계산 컬럼 추가 -> 분 단위 집계 -> Parquet로 저장
    # COPY (...) TO 'output.parquet' (FORMAT PARQUET); 형태로 바로 저장
    # read_json_auto() 함수로 JSONL 파일 로드
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

    # 검증
    row = con.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{out}')").fetchone()[0]
    print(f"Wrote: {OUT} rows={row}")

    con.close()

if __name__ == "__main__":
    main()