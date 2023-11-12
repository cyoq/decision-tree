import pandas as pd

from decision_tree import DecisionTree


def load_train_data(csv_file: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(csv_file, sep=sep)


def main():
    # df = load_train_data("./dataset/example.csv")
    # class_name = "Play"
    # feature_names = ["Outlook", "Temperature", "Humidity", "Windy"]

    df = load_train_data("./dataset/id3.csv")
    class_name = "infected_with_X_disease"
    feature_names = ["fever", "cough", "short_of_breath"]

    tree = DecisionTree(df, class_name, feature_names, "Yes", "No")
    tree.train()
    print(tree.root)
    print(tree.mermaid_diagram_str())
    # tree.create_mermaid_md("new_diagram.md")
    # print(
    #     tree.predict(
    #         {
    #             "Outlook": "Sunny",
    #             "Humidity": "High",
    #             "Temperature": "Low",
    #             "Windy": True,
    #         }
    #     )
    # )
    # 10 predictions of a probability node
    for _ in range(10):
        print(
            tree.predict(
                {
                    "fever": "No",
                    "short_of_breath": "Yes",
                    "cough": "Yes",
                }
            )
        )


if __name__ == "__main__":
    main()
