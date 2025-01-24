import asyncio
from pathlib import Path

import polars as pl
from logfire.experimental.query_client import AsyncLogfireQueryClient
from loguru import logger

FILE_NAME = "knd_evals_traces.parquet"

query = """
WITH agent_traces AS (
  SELECT DISTINCT trace_id 
  FROM records 
  WHERE attributes->>'agent_name' = 'knd_evals'
  AND end_timestamp >= NOW() - INTERVAL '10 minutes'
)
SELECT 
  r.trace_id,
  r.span_id,
  r.start_timestamp,
  r.end_timestamp,
  r.duration,
  r.level,
  r.message,
  r.tags,
  r.attributes->>'agent_name' as agent_name
FROM records r
JOIN agent_traces at ON r.trace_id = at.trace_id
ORDER BY r.trace_id, r.start_timestamp;
"""


async def get_df() -> pl.DataFrame:
    async with AsyncLogfireQueryClient(read_token="H0CTvcy0WCrl6xjxm8r8ZjWxP3LPSq5Mzdv81GvXXRPz") as client:
        result = await client.query_arrow(sql=query)
    return pl.DataFrame(result)


async def save_df():
    df = await get_df()
    df.write_parquet(FILE_NAME)


async def main():
    if not Path(FILE_NAME).exists():
        await save_df()
    df = pl.read_parquet(FILE_NAME)

    logger.info(df)

    total_runs = df["trace_id"].n_unique()
    logger.info(f"Total runs: {total_runs}")

    res = (
        df.filter(pl.col("tags").list.contains("yoo_check"))
        .group_by("trace_id", maintain_order=True)
        .agg(pl.len().alias("num_check_errors"))
    )

    logger.info(res)

    check_accuracy = (total_runs - res["num_check_errors"].sum()) / total_runs
    logger.info(f"Check accuracy: {check_accuracy}")

    res = df.filter(pl.col("level") == 17)
    logger.info(res)


if __name__ == "__main__":
    asyncio.run(main())
