from project import mermaid


def test_mermaid():
    hello = mermaid.Node(value="hello", shape=mermaid.NodeShape.SQUARE)
    world = mermaid.Node(value="world")
    to_you = mermaid.Node(value="to you")

    nodes = [hello, world, to_you]

    links = [
        mermaid.Link(hello, world, text="Yes"),
        mermaid.Link(hello, to_you, text="No"),
    ]

    diagram = mermaid.MermaidDiagram(nodes, links, title="Hello World")

    expected = """
---
title: Hello World
---
graph LR
    hello["hello"]
    world("world")
    to_you("to you")

    hello --|Yes|--> world
    hello --|No|--> to_you
"""

    print(diagram.draw())

    assert diagram.draw() == expected
