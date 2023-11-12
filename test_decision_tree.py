import pytest
import pandas as pd

from project.decision_tree import DecisionTree, VisitedFeatures


@pytest.fixture(
    scope="module",
)
def df() -> pd.DataFrame:
    csv_file = "example.csv"
    df = pd.read_csv(csv_file, sep=";")
    yield df


@pytest.fixture(
    scope="module",
)
def tree(df: pd.DataFrame) -> DecisionTree:
    tree = DecisionTree(
        df, "Play", ["Outlook", "Temperature", "Humidity", "Windy"], "Yes", "No"
    )
    yield tree


class TestDecisionTree:
    @pytest.mark.parametrize(
        "visited_features, expected_output",
        [
            ({}, ""),
            ({"Outlook": "Sunny"}, "Outlook == 'Sunny'"),
            (
                {"Outlook": "Sunny", "Temperature": "Hot"},
                "Outlook == 'Sunny' & Temperature == 'Hot'",
            ),
            (
                {"Outlook": "Sunny", "Windy": True},
                "Outlook == 'Sunny' & Windy == True",
            ),
        ],
    )
    def test_create_filter(
        self,
        tree: DecisionTree,
        visited_features: VisitedFeatures,
        expected_output: str,
    ):
        assert tree._create_filter(visited_features) == expected_output
