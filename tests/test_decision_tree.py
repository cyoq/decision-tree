import pytest
import pandas as pd

import project.decision_tree as dt


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
def tree(df: pd.DataFrame) -> dt.DecisionTree:
    tree = dt.DecisionTree(
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
        tree: dt.DecisionTree,
        visited_features: dt.VisitedFeatures,
        expected_output: str,
    ):
        tree.visited_features = visited_features
        assert tree._create_filter() == expected_output
        tree.clear()

    def test_create_tree(self, tree: dt.DecisionTree):
        expected_tree = dt.Node(
            label="Outlook",
            children={
                "Sunny": dt.Node(
                    label="Humidity",
                    children={
                        "High": dt.Node(label="Play", children=None, answer="No"),
                        "Normal": dt.Node(label="Play", children=None, answer="Yes"),
                    },
                    answer=None,
                ),
                "Overcast": dt.Node(label="Play", children=None, answer="Yes"),
                "Rainy": dt.Node(
                    label="Windy",
                    children={
                        False: dt.Node(label="Play", children=None, answer="Yes"),
                        True: dt.Node(label="Play", children=None, answer="No"),
                    },
                    answer=None,
                ),
            },
            answer=None,
        )
        tree.train()
        assert tree.root == expected_tree
