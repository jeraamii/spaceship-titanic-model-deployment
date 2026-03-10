import pickle

from data_ingestion import load_data
from pre_processing import build_preprocessor
from train import train_model
from evaluation import evaluate_model


def run_pipeline():

    df = load_data()

    y = df["Transported"]
    X = df.drop("Transported", axis=1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

    if "Name" in X.columns:
        X = X.drop("Name", axis=1)

    preprocessor = build_preprocessor(X)

    model = train_model(X_train, y_train, preprocessor)

    evaluate_model(model, X_test, y_test)

    pickle.dump(model, open("artifacts/model.pkl","wb"))
    pickle.dump(preprocessor, open("artifacts/preprocessor.pkl","wb"))

    print("Model and Preprocessor saved!")


if __name__ == "__main__":
    run_pipeline()