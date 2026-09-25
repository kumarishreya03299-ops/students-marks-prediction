import pandas as pd
import joblib
df = pd.read_csv("Student_Marks.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)

x=df[["number_courses","time_study"]]
y=df["Marks"]

print("features (X):")
print(x.head())

print("target (Y):")
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x , y ,test_size=0.2 ,random_state=42 )
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model,"model.pkl")
print("Model training completed!")

prediction = model.predict([[3, 5]])
print("Predicted Marks:", prediction[0])

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R² Score:", r2)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()