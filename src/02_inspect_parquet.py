import pyarrow.parquet as pq
from pathlib import Path

DATA = Path("data/agg/minute_metrics.parquet")

parquet_file = pq.ParquetFile(DATA)

# 메타데이터 출력
metadata = parquet_file.metadata
print(f"Number of rows: {metadata.num_rows}")
print(f"Number of columns: {metadata.num_columns}")
print(f"Number of row groups: {metadata.num_row_groups}")
print()
# Number of rows: 60480
# Number of columns: 6
# Number of row groups: 1

# 스키마 (컬럼 이름과 타입) 출력
schema = parquet_file.schema
for i in range(len(schema)):
    field = schema[i]
    print(f" {field.name} : {field.physical_type}")
print()
# ts : INT64
# service : BYTE_ARRAY
# req_count : INT64
# err_count : INT64
# error_rate : DOUBLE
# p95_latency_ms : INT64

# 컬럼별 압축 정보 출력
for i in range(metadata.num_row_groups):
    row_group = metadata.row_group(i)
    print(f"Row Group {i}:")
    for j in range(row_group.num_columns):
        column = row_group.column(j)
        ratio = column.total_compressed_size / column.total_uncompressed_size if column.total_uncompressed_size > 0 else 0
        print(f" {column.path_in_schema} : {ratio:.2f}배 압축")

# Row Group 0:
# ts : 0.38배 압축
# service : 0.05배 압축
# req_count : 0.98배 압축
# err_count : 0.51배 압축
# error_rate : 0.60배 압축
# p95_latency_ms : 0.98배 압축
