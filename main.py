import pandas as pd

from decision_tree import DecisionTree


def load_train_data(csv_file: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(csv_file, sep=sep)


def main():
    df = load_train_data("./example.csv")
    class_name = "Play"
    feature_names = ["Outlook", "Temperature", "Humidity", "Windy"]
    tree = DecisionTree(df, class_name, feature_names, "Yes", "No")
    tree.train()
    print(tree.root)


if __name__ == "__main__":
    main()
