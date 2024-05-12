from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ray import tune, data
import time


def train_model(config, data):

    train_df = data.to_pandas()

    x_train = train_df.drop(columns=["Cover_Type"])
    y_train = train_df["Cover_Type"]
    # train Random forest

    clf = RandomForestClassifier()  # default hyperparameters
    print("Default Hyper Parameters:\n", clf.get_params())

    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=123)
    cv_scores = cross_val_score(clf,
                                x_train,
                                y_train,
                                cv=kfold,
                                scoring="accuracy"
                                )
    score = cv_scores.mean()

    return {"score": score}


search_space = {
        "n_estimators": tune.grid_search([50, 200]),
        "max_depth": tune.grid_search([1, 4, 10]),
        "ccp_alpha": tune.grid_search([0, .5]),
        }

# load data
train_dataset = data.read_csv("./train.csv")
print("Schema:\n", train_dataset.schema())

# trainable_with_resources = tune.with_resources(trainable, {"cpu": 0.5})

tuner = tune.Tuner(
        tune.with_parameters(train_model, data=train_dataset),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="score", mode="max"
            ),
        # trainable_with_resources
        )
start_time = time.time()
results = tuner.fit()
end_time = time.time()
elapsed = end_time - start_time

print("Best HP:\n", results.get_best_result())
print("----------------")
print("Results:\n", results.get_dataframe())
print("-------TIME---------")
print(f"Parameter tuning took {elapsed:.6f} seconds")
