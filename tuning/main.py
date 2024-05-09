from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import time

# load data
train_df = pd.read_csv("data/train.csv")

x_train = train_df.drop(columns=["Cover_Type"])
y_train = train_df["Cover_Type"]
# train Random forest

clf = RandomForestClassifier()  # default hyperparameters
print("Default Hyper Parameters:\n", clf.get_params())

start_time = time.time()
k = 10
kfold = KFold(n_splits=k, shuffle=True, random_state=123)
cv_scores = cross_val_score(clf,
                            x_train,
                            y_train,
                            cv=kfold,
                            scoring="accuracy"
                            )
end_time = time.time()

print("Accuracy:", cv_scores.mean())
elapsed = end_time - start_time
print(f"CV took {elapsed:.6f} seconds")
