from pathlib import Path
import duckdb

DB = Path("data/analytics.duckdb")
PARQUET = "data/processed/telco_clean.parquet"

con = duckdb.connect(DB.as_posix())
con.execute(f"CREATE OR REPLACE TABLE customers AS SELECT * FROM read_parquet('{PARQUET}')")

print(con.sql("SELECT COUNT(*) AS customers FROM customers"))
print(con.sql("SELECT Churn, COUNT(*) AS n FROM customers GROUP BY Churn"))
con.close()