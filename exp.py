import pytest

from joke_testing_utils import judge_joke

jokes_dict = {
    "Why don't scientists trust atoms? Because they make up everything!": True,
    "What did the grape say when it got stepped on? Nothing, it just let out a little wine!": True,
    "Why did the scarecrow win an award? Because he was outstanding in his field!": True,
    "What's brown and sticky? A stick.": False,
    "What do you call a bear with no teeth? A gummy bear.": False,
}


@pytest.mark.parametrize("joke", jokes_dict.keys())
def test_joke(joke, results_bag):
    judgement = judge_joke(joke, jokes_dict)
    results_bag.judgement = judgement
    assert judgement


def test_synthesis(module_results_df):
    print("\n   `module_results_df` dataframe:\n")
    print(module_results_df)
    module_results_df.to_csv("module_results.csv")


if __name__ == "__main__":
    res = pytest.main(["-s", "-v", __file__])
    print(res)
