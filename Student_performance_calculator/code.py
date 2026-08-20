import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data={
    "study_hours":[1,2,3,4,5,6,7,8,9,10],
    "attendance":[55,60,65,70,72,78,82,88,92,95],
    "previous_marks":[45,50,55,60,62,68,72,78,85,90],
    "assignments":[3,4,5,5,6,7,7,8,9,10],
    "final_marks":[48,52,58,63,66,72,76,82,88,94]
}
# creating dataframe:
df=pd.DataFrame(data)
print("\n======DataFrame======\n")
print(df)

# checking missing values
print("\n=====Missing Values=====\n")
print(df.isnull().sum())

# feature selection
x=df[["study_hours","attendance","previous_marks","assignments"]]
y=df["final_marks"]

print("\n====features===\n")
print(x)

print("\n=====target====\n")
print(y)

# split data into training and testingsets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("\n ====Training data====\n")
print(x_train)
print("\n====testing====\n")
print(x_test)

# create machine learning model
model=LinearRegression()

# training the model
model.fit(x_train,y_train)
print("\n==========model_trained Successfully====")

# make predictions
predictions=model.predict(x_test)
print("\n ===predictions==\n")

for actual,preducted in zip(y_test,predictions):
    print(
        f"actual: {actual:.2f}, predicted: {preducted:.2f}"
    )
# evaluate model
mae=mean_absolute_error(y_test,predictions)
mse=mean_squared_error(y_test,predictions)  
r2=r2_score(y_test,predictions)

print("\n===== MODEL PERFORMANCE =====")

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# predict for a new student
new_student=pd.DataFrame({
    "study_hours":[6],
    "attendance":[90],
    "previous_marks":[80],
    "assignments":[9]
})
prediction=model.predict(new_student)
print(new_student)
print(f"\nPredicted final marks:"
      f"{prediction[0]:.2f}")

