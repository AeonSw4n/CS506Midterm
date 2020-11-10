
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

X_train = pd.read_csv("./data/X_train.csv")
y_train = X_train['Score']

X = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score', 'Time'])

predictionSet = pd.read_csv("./data/prediction.csv")
X_predict = predictionSet.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score', 'Time'])

model = KNeighborsClassifier(n_neighbors=5).fit(X, y_train)
predictionSet['Score'] = model.predict(X_predict)

submission = predictionSet[['Id', 'Score']]
print(submission.head())
submission.to_csv("./data/submission.csv", index=False)