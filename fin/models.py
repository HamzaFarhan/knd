from enum import StrEnum

from pydantic import BaseModel, Field


class InvestmentGoal(StrEnum):
    RETIREMENT = "Retirement Planning"
    WEALTH_BUILDING = "Wealth Building"
    SPECIFIC_GOAL = "Specific Financial Goal"
    PASSIVE_INCOME = "Passive Income Generation"
    OTHER = "Other"


class InvestmentExperience(StrEnum):
    NO_EXPERIENCE = "No Experience"
    SOME_EXPERIENCE = "Some Experience"
    REGULAR_INVESTOR = "Regular Investor"
    PROFESSIONAL = "Professional"


class InvestmentCapacity(StrEnum):
    LESS_THAN_10 = "Less than 10%"
    TEN_TO_TWENTY_FIVE = "10-25%"
    TWENTY_FIVE_TO_FIFTY = "25-50%"
    MORE_THAN_FIFTY = "More than 50%"


class LossReaction(StrEnum):
    SELL_EVERYTHING = "Sell everything immediately"
    SELL_SOME_POSITIONS = "Sell some positions"
    HOLD_AND_WAIT = "Hold and wait"
    BUY_MORE_AT_LOWER_PRICES = "Buy more at lower prices"


class InvestmentType(StrEnum):
    STOCKS = "Stocks"
    MUTUAL_FUNDS = "Mutual Funds"
    ETFs = "ETFs"
    COMMODITIES = "Commodities"
    BONDS = "Bonds"
    REAL_ESTATE = "Real Estate"
    OTHERS = "Others"


class InvestmentTimeline(StrEnum):
    LESS_THAN_2_YEARS = "Less than 2 years"
    TWO_TO_FIVE_YEARS = "2-5 years"
    FIVE_TO_TEN_YEARS = "5-10 years"
    MORE_THAN_TEN_YEARS = "More than 10 years"


class UserProfile(BaseModel):
    goal: InvestmentGoal = Field(
        description="""
Question to ask: 'What brings you to explore investment opportunities today?'.
'Specific Financial Goal' could be home, education, etc.
""".strip()
    )
    other_goal: str = Field(
        default="", description="If you define the user's goal as 'Other', ask for more details."
    )
    experience: InvestmentExperience = Field(
        description="""
Question to ask: 'How would you describe your investment experience?'.
'Some Experience' could be with basic investments (mutual funds, stocks, etc.)
""".strip()
    )
    income_range: str = Field(description="Question to ask: 'What is your current annual income range?'")
    investment_capacity: InvestmentCapacity = Field(
        description="Question to ask: 'What percentage of your monthly income can you comfortably set aside for investments? (<10%, 10-25%, 25-50%, >50%)'"
    )
    loss_reaction: LossReaction = Field(
        description="Question to ask: 'How would you react if your investment lost 20% of its value in one month?'"
    )
    investment_types: list[InvestmentType] = Field(
        description="""
Question to ask: 'Which investment types interest you? (Select all that apply)'.
- Stocks
- Mutual Funds
- ETFs
- Commodities
- Bonds
- Real Estate
- Others (specify)
""".strip()
    )
    other_investment_types: str = Field(
        default="", description="If one of the investment types is defined as 'Others', ask for more details."
    )
    investment_timeline: InvestmentTimeline = Field(
        description="Question to ask: 'When do you expect to need most of this invested money?'"
    )
