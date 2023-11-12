from dataclasses import dataclass
import enum

# Inspired by examples of https://pypi.org/project/python-mermaid/


class Direction(enum.Enum):
    LR = "LR"
    TB = "TB"


class NodeShape(enum.Enum):
    ROUND = {"open": "(", "close": ")"}
    SQUARE = {"open": "[", "close": "]"}

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


@dataclass
class Link:
    _from: Node
    to: Node
    text: str | None


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
            node_str += f'{" " * 4}{node.value.replace(" ", "_")}{node.shape.open}"{node.value}"{node.shape.close}\n'
        return node_str

    def _aggregate_links(self) -> str:
        link_str = ""
        for link in self.links:
            _from = link._from.value.replace(" ", "_")
            to = link.to.value.replace(" ", "_")

            arrow = "-->"
            if link.text is not None:
                arrow = f"--|{link.text}|-->"

            link_str += f'{" " * 4}{_from} {arrow} {to}\n'
        return link_str
