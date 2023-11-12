from dataclasses import dataclass
from typing import NewType, Union
import numpy as np
import pandas as pd

import mermaid


@dataclass
class Attribute:
    label: str
    positives: int  # positive data size
    negatives: int  # negative data size
    data_size: int  # feature data size
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
    # dict[str, int] is for probabalistic model
    answer: str | dict[str, int] | None

    def is_probabalistic(self) -> bool:
        return isinstance(self.answer, str)


VisitedFeatures = dict[str, Union[str, bool]]


class DecisionTree:
    def __init__(
        self,
        train_data: pd.DataFrame,
        class_name: str,
        feature_names: list[str],
        positive_label: str,
        negative_label: str,
        depth_threshold: int = 10,
    ):
        self.df = train_data
        self.class_name = class_name
        self.feature_names = feature_names
        self.positive_label = positive_label
        self.negative_label = negative_label
        # Contains a dict of feature: attribute
        self.visited_features: VisitedFeatures | None = None
        self.root: Node | None = None
        self.depth_threshold = depth_threshold

    def train(self):
        """
        1. Calculate overall entropy
        2. Calculate entropy for each feature:
        2.1. Calculate entropy for each unique label
        2.2. Get an average entropy for the feature
        3. Calculate the gain
        4. Choose a feature with max gain
        5. Repeat the process with the child nodes
        """
        self.root = self._generate_subtree(self.feature_names)

    def mermaid_diagram(self) -> str:
        """
        Create a Mermaid diagram of a tree
        """
        mermaid_nodes: set[mermaid.Node] = set()
        mermaid_links: set[mermaid.Link] = set()

        self._prepare_mermaid_data(self.root, mermaid_nodes, mermaid_links)

        diagram = mermaid.MermaidDiagram(
            mermaid_nodes, mermaid_links, direction=mermaid.Direction.TB
        )

        return diagram.draw()

    def clear(self):
        self.root = None
        self.visited_features = None

    def _prepare_mermaid_data(
        self,
        node: Node,
        mermaid_nodes: set[mermaid.Node],
        mermaid_links: set[mermaid.Link],
    ):
        if node.children is None:
            return

        mermaid_node = self._create_mermaid_node(node)
        mermaid_nodes.add(mermaid_node)

        for link_text, child in node.children.items():
            child_node = self._create_mermaid_node(child)
            mermaid_nodes.add(child_node)
            mermaid_links.add(mermaid.Link(mermaid_node, child_node, link_text))
            self._prepare_mermaid_data(child, mermaid_nodes, mermaid_links)

    def _create_mermaid_node(self, node: Node) -> mermaid.Node:
        if node.answer is not None:
            return mermaid.Node(
                value=f"Play?: {node.answer}", shape=mermaid.NodeShape.SQUARE
            )

        if node.is_probabalistic():
            probabilities = ",".join([f"{k}: {v}" for k, v in node.answer.items()])
            return mermaid.Node(
                value=f"Play?: {probabilities}",
                shape=mermaid.NodeShape.STADIUM,
            )

        return mermaid.Node(value=node.label)

    def _generate_subtree(
        self,
        feature_names: list[str],
        depth: int = 0,
    ) -> Node | None:
        # if we get no feature names, then something is wrong
        # and we should use probabalistic form
        if len(feature_names) == 0:
            return None

        if depth > self.depth_threshold:
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

                sub_tree = self._generate_subtree(
                    investigated_feature_names,
                    depth + 1,
                )

                # We should use a probabalistic form
                # There must be a conflict and no 100% answer
                if sub_tree is None:
                    probabalistic_node: Node = self._create_probabalistic_node(
                        max_gain_feature
                    )
                    children[attr.label] = probabalistic_node
                else:
                    children[attr.label] = sub_tree
            # After we took a look at one feature attribute, we remove it to replace it with other attribute
            # For example, we took a look at {outlook: sunny}, then we remove it to replace with {outlook: rain}
            del self.visited_features[max_gain_feature]

        sub_tree = Node(label=max_gain_feature, children=children, answer=None)

        return sub_tree

    def _create_probabalistic_node(self, prob_feature: str) -> Node:
        # We use visited_features to recereate a sub tree with the probabalistic view
        mask = " & ".join(
            [
                f"{feature} == {self._do_quotes(feature, attr)}"
                for feature, attr in self.visited_features.items()
                if feature != prob_feature
            ]
        )
        probabilities: dict[str, int] = (
            self.df.query(mask)[self.class_name].value_counts().to_dict()
        )
        return Node(label=self.class_name, children=None, answer=probabilities)

    def _get_answer(self, feature: str, attribute: str) -> str:
        categories = self.df.query(f"{self._create_filter()}")[self.class_name].unique()

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
            data_size = len(self.df)
            positives = len(self.df[self.df[self.class_name] == self.positive_label])
        else:
            data_size = len(self.df.query(self._create_filter()))
            positives = len(
                self.df.query(
                    f"{self._create_filter()} & "
                    f"{self.class_name} == '{self.positive_label}'"
                )
            )

        negatives = data_size - positives
        h = entropy(positives, negatives, data_size)

        for name, attributes in features.items():
            gains[name] = h - self._calculate_avg_feature_entropy(attributes)

        return gains

    def _do_quotes(self, feature: str, attribute: Union[str, bool, int]) -> str:
        """
        Pandas query engine requires strings to be in quotes, but
        ints and booleans without quotes. Strings have Object dtype.
        """
        typ = self.df[feature].dtype
        if typ == "O":
            return f"'{attribute}'"
        return f"{attribute}"

    def _create_filter(self) -> str:
        """
        A filter for the visited features
        """
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
            data_size = len(self.df)
        else:
            data_size = len(self.df.query(self._create_filter()))

        gain = 0.0

        for attribute in attributes:
            gain += (
                (attribute.positives + attribute.negatives)
                / data_size
                * attribute.entropy
            )
        return gain

    def _get_feature_attributes(self, feature_name: str) -> list[Attribute]:
        name = feature_name
        features: list[Attribute] = list()

        # Get unique attributes from a feature column
        for attribute in self.df[name].unique():
            # If it is a root, then calculate entropy over all dataset
            if len(self.visited_features) == 0:
                data_size = len(self.df[self.df[name] == attribute])

                positives = len(
                    self.df[
                        (self.df[name] == attribute)
                        & (self.df[self.class_name] == self.positive_label)
                    ]
                )
            # If it is not a root, calculate entropy of a sub tree
            else:
                data_size = len(
                    self.df.query(
                        f"{name} == {self._do_quotes(name, attribute)} & "
                        f"{self._create_filter()}"
                    )
                )

                positives = len(
                    self.df.query(
                        f"{self._create_filter()} & "
                        f"{name} == {self._do_quotes(name, attribute)} & "
                        f"{self.class_name} == {self._do_quotes(self.class_name, self.positive_label)}"
                    )
                )

            # Ignore attributes with no values
            if data_size == 0:
                continue

            negatives = data_size - positives

            h_f = entropy(positives, negatives, data_size)

            features.append(
                Attribute(
                    label=attribute,
                    positives=positives,
                    negatives=negatives,
                    data_size=data_size,
                    entropy=h_f,
                )
            )

        return features
