import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop unnecessary columns
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Convert gender to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-hot encoding for Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    return df


def split_features(df):

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y