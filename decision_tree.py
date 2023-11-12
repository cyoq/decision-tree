from dataclasses import dataclass
from typing import NewType, Union
import numpy as np
import pandas as pd


@dataclass
class Attribute:
    label: str
    p: int  # positive example size
    n: int  # negative example size
    pn: int  # feature size
    entropy: float

    def is_pure(self) -> bool:
        return np.equal(self.entropy, 0.0)


def entropy(p: int, n: int, pn: int) -> float:
    return -(
        p / pn * np.log2(p / pn, where=(p != 0))
        + n / pn * np.log2(n / pn, where=(n != 0))
    )


Edge = NewType("Edge", str)


@dataclass
class Node:
    label: str
    children: dict[Edge, "Node"] | None
    answer: str | None


VisitedFeatures = dict[str, Union[str, bool]]


class DecisionTree:
    def __init__(
        self,
        train_data: pd.DataFrame,
        class_name: str,
        feature_names: list[str],
        positive_label: str,
        negative_label: str,
    ):
        self.df = train_data
        self.class_name = class_name
        self.feature_names = feature_names
        self.positive_label = positive_label
        self.negative_label = negative_label
        # Contains a dict of feature: attribute
        self.visited_features: VisitedFeatures | None = None
        self.root: Node | None = None

    def train(self):
        # 1. Calculate overall entropy
        # 2. Calculate entropy for each feature:
        #   2.1. Calculate entropy for each unique label
        #   2.2. Get an average entropy for the feature
        # 3. Calculate the gain
        # 4. Choose a feature with max gain
        # 5. Repeat the process with the child nodes
        self.root = self._generate_subtree(self.feature_names)

    def clear(self):
        self.root = None
        self.visited_features = None

    def _generate_subtree(
        self,
        feature_names: list[str],
        depth: int = 0,
    ) -> Node | None:
        if len(feature_names) == 0:
            return None

        if depth > 10:
            return None

        if self.visited_features is None:
            self.visited_features = {}

        features: dict[str, list[Attribute]] = {}
        for feature in feature_names:
            features[feature] = self._get_feature_attributes(feature)

        gains = self._calculate_gains(features)
        max_gain_feature: str = max(gains, key=gains.get)

        children = {}

        for attr in features[max_gain_feature]:
            self.visited_features[max_gain_feature] = attr.label

            if attr.is_pure():
                children[attr.label] = Node(
                    label=self.class_name,
                    children=None,
                    answer=self._get_answer(max_gain_feature, attr.label),
                )
            else:
                investigated_feature_names = [
                    feature_name
                    for feature_name in feature_names
                    if feature_name
                    not in [feature for feature in self.visited_features.keys()]
                ]

                children[attr.label] = self._generate_subtree(
                    investigated_feature_names,
                    depth + 1,
                )

            # After we took a look at one feature attribute, we remove it to replace it with other attribute
            # For example, we took a look at {outlook: sunny}, then we remove it to replace with {outlook: rain}
            del self.visited_features[max_gain_feature]

        sub_tree = Node(label=max_gain_feature, children=children, answer=None)

        return sub_tree

    def _get_answer(self, feature: str, attribute: str) -> str:
        categories = self.df.query(
            f"{self._create_filter()}"
            # f"{self.df[feature]} == '{attribute}'"
        )[self.class_name].unique()

        if len(categories) > 1:
            raise ValueError(
                f"For some reason attribute is not 100% pure. Feature {feature}, "
                f"attribute {attribute}, categories: {categories}"
            )
        return categories[0]

    def _calculate_gains(
        self, features: dict[str, list[Attribute]]
    ) -> dict[str, float]:
        gains = {name: 0.0 for name in features.keys()}

        if len(self.visited_features) == 0:
            pn = len(self.df)
            p = len(self.df[self.df[self.class_name] == self.positive_label])
        else:
            pn = len(self.df.query(self._create_filter()))
            p = len(
                self.df.query(
                    f"{self._create_filter()} & "
                    f"{self.class_name} == '{self.positive_label}'"
                )
            )

        n = pn - p
        h = entropy(p, n, pn)

        for name, attributes in features.items():
            gains[name] = h - self._calculate_avg_feature_entropy(attributes)

        return gains

    def _do_quotes(self, feature: str, attribute: Union[str, bool]) -> str:
        typ = self.df[feature].dtype
        if typ == "O":
            return f"'{attribute}'"
        return f"{attribute}"

    def _create_filter(self) -> str:
        if len(self.visited_features) == 0:
            return ""

        return " & ".join(
            [
                f"{feature} == {self._do_quotes(feature, attr)}"
                for feature, attr in self.visited_features.items()
            ]
        )

    def _calculate_avg_feature_entropy(self, attributes: list[Attribute]) -> float:
        if len(self.visited_features) == 0:
            pn = len(self.df)
        else:
            pn = len(self.df.query(self._create_filter()))

        gain = 0.0

        for attribute in attributes:
            gain += (attribute.p + attribute.n) / pn * attribute.entropy
        return gain

    def _get_feature_attributes(self, feature_name: str) -> list[Attribute]:
        name = feature_name
        features: list[Attribute] = list()

        # Get unique values from a feature column
        for attribute in self.df[name].unique():
            if len(self.visited_features) == 0:
                pn = len(self.df[self.df[name] == attribute])

                p = len(
                    self.df[
                        (self.df[name] == attribute)
                        & (self.df[self.class_name] == self.positive_label)
                    ]
                )
            else:
                pn = len(
                    self.df.query(
                        f"{name} == {self._do_quotes(name, attribute)} & "
                        f"{self._create_filter()}"
                    )
                )

                p = len(
                    self.df.query(
                        f"{self._create_filter()} & "
                        f"{name} == {self._do_quotes(name, attribute)} & "
                        f"{self.class_name} == {self._do_quotes(self.class_name, self.positive_label)}"
                    )
                )

            # Ignore attribute with no values
            if pn == 0:
                continue

            n = pn - p

            h_f = entropy(p, n, pn)

            features.append(Attribute(label=attribute, p=p, n=n, pn=pn, entropy=h_f))

        return features
