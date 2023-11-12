import pytest
import pandas as pd

from project.decision_tree import DecisionTree, Node, VisitedFeatures


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
        tree.visited_features = visited_features
        assert tree._create_filter() == expected_output
        tree.clear()

    def test_create_tree(self, tree: DecisionTree):
        expected_tree = Node(
            label="Outlook",
            children={
                "Sunny": Node(
                    label="Humidity",
                    children={
                        "High": Node(label="Play", children=None, answer="No"),
                        "Normal": Node(label="Play", children=None, answer="Yes"),
                    },
                    answer=None,
                ),
                "Overcast": Node(label="Play", children=None, answer="Yes"),
                "Rainy": Node(
                    label="Windy",
                    children={
                        False: Node(label="Play", children=None, answer="Yes"),
                        True: Node(label="Play", children=None, answer="No"),
                    },
                    answer=None,
                ),
            },
            answer=None,
        )
        tree.train()
        assert tree.root == expected_tree
