from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def train_model(X, y, preprocessor):

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        "model__C":[0.01,0.1,1,10]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X,y)

    return grid.best_estimator_