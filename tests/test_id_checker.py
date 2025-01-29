import pytest

import polars as pl

from agent import IdCheckerAgent



test_df = pl.read_csv("id_checker_test_data.csv")


id_ex

@pytest.mark.parametrize("row", test_df.iter_rows())
def test_id_checker(row, results_bag):
    gen_id = IdCheckerAgent.generate_id(row["text"])
    assert gen_id == row["id"]


def test_synthesis(module_results_df):
    print("\n   `module_results_df` dataframe:\n")
    print(module_results_df)
    module_results_df.to_csv("module_results.csv")


if __name__ == "__main__":
    res = pytest.main(["-s", "-v", __file__])
    print(res)