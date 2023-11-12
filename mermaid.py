from dataclasses import dataclass
import enum

# Inspired by examples of https://pypi.org/project/python-mermaid/


class Direction(enum.Enum):
    LR = "LR"
    TB = "TB"


class NodeShape(enum.Enum):
    ROUND = {"open": "(", "close": ")"}
    SQUARE = {"open": "[", "close": "]"}
    STADIUM = {"open": "([", "close": "])"}

    @property
    def open(self) -> str:
        return self.value["open"]

    @property
    def close(self) -> str:
        return self.value["close"]


@dataclass
class Node:
    value: str
    shape: NodeShape = NodeShape.ROUND

    def __hash__(self):
        return hash((self.value, self.shape.name))


@dataclass
class Link:
    from_: Node
    to: Node
    text: str | None

    def __hash__(self):
        return hash((self.from_, self.to))


class MermaidDiagram:
    def __init__(
        self,
        nodes: list[Node],
        links: list[Link],
        direction: Direction = Direction.LR,
        title: str | None = None,
    ):
        self.nodes = nodes
        self.links = links
        self.title = title
        self.direction = direction

    def draw(self) -> str:
        title = ""
        if self.title is not None:
            title = f"""
---
title: {self.title}
---"""

        nodes = self._aggregate_nodes()
        links = self._aggregate_links()

        return f"""{title}
graph {self.direction.value}
{nodes}
{links}"""

    def _aggregate_nodes(self) -> str:
        node_str = ""
        for node in self.nodes:
            node_str += f'{" " * 4}{self._fix_name(node.value)}{node.shape.open}"{node.value}"{node.shape.close}\n'
        return node_str

    def _aggregate_links(self) -> str:
        link_str = ""
        for link in self.links:
            from_ = self._fix_name(link.from_.value)
            to = self._fix_name(link.to.value)

            arrow = "-->"
            if link.text is not None:
                arrow = f"--|{link.text}|-->"

            link_str += f'{" " * 4}{from_} {arrow} {to}\n'
        return link_str

    def _fix_name(self, value: str) -> str:
        # TODO: use regex
        return (
            value.replace(" ", "_")
            .replace("?", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace("'", "_")
            .replace(":", "_")
            .replace(",", "_")
        )
