from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ray import tune
import pandas as pd


def train_model(config):

    # load data
    train_df = pd.read_csv("data/train.csv")

    x_train = train_df.drop(columns=["Cover_Type"])
    y_train = train_df["Cover_Type"]
    # train Random forest

    clf = RandomForestClassifier()  # default hyperparameters
    print("Default Hyper Parameters:\n", clf.get_params())

    k = 10
    kfold = KFold(n_splits=k, shuffle=True, random_state=123)
    cv_scores = cross_val_score(clf,
                                x_train,
                                y_train,
                                cv=kfold,
                                scoring="accuracy"
                                )

    print("Accuracy:", cv_scores.mean())
    tune.report(accuracy=cv_scores.mean())


search_space = {
        "n_estimators": tune.randint(50, 200),
        "max_depth": tune.randint(1, 10),
        "min_samples_split": tune.uniform(0.01, 0.1),
        }

analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=20,
        resources_per_trial={"cpu": 0.5, "gpu": 0},
        )

best = analysis.best_config
print("Best HP:\n", best)
print("----------------")
print("Results:\n", analysis.results)
