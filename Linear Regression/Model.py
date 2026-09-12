#Step 1 import model/tools/libraries
import xgboost as xgb        #this imports the ml model
from sklearn.model_selection import train_test_split #we dont import the complete sklearn because it too big hence we only import the tools we will use.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # this tells us how accurate our predicitons are compared to the real ans and how how far my predictions are
from sklearn.preprocessing import LabelEncoder     #this helps us to convert out text values to numeric values for regression
import pandas as pd # this is used to process/parse csv/excel files
import matplotlib.pyplot as plt # this is a library that helps us visualize/plot our predicted results, "plt" is just a short cut we will be using in the code ahead.


#Step 2 Load/Explore/Understand Data
df = pd.read_excel('data.xlsx')
#print(df)
#print(df.dtypes)
#print(df.head())                                            #If you want to see the first few rows

#print(df.head(10))                                          #If you want to see the first 10 rows

#print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")       # See how big the data is

#print(df.columns)                                           # See all column names

#print(df.info())                                             # See detailed info such as data types and memory usage

#print(df.duplicated())                                       # See duplicates


#Step 3 Clean & Prepare data
df = df.dropna()                            # drop rows that have missing values
df = df.drop_duplicates()                   # drop rows that have duplicates

Text_Columns = ['File', 'Base Unit', 'PartNo', 'Machine', 'Material_Code', 'Material_Group']

for col in Text_Columns:                         #this loop is used to transform all the text data into numeric data since we are doing regression
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print(df.dtypes)                                 #this shows us if it actually transformed everything correctly
print("Data Cleaned")

#Step 4 Split Data for training
# separate features (X) and target (y)
X = df.drop('Processing_Time_Seconds', axis=1)  # This will process all columns except 'Processing_Time_Seconds' which is target column
y = df['Processing_Time_Seconds']               #() = function call (does an action) [] = accessing/indexing (gets something)
print(f"Features (X) shape:, {X.shape}")         # we use this to make sure no rows and columns were missed
print(f"Target (y) shape:, {y.shape}")           # we use this to make sure only 1 selected target column shows

# Now split into train (80%) and train (20%) data
X_train, X_test, y_train, y_test = train_test_split(   #Think of this like splitting your data in 4 pieces
    X, y,
    test_size = 0.2,                               
    random_state = 42                                  
    )

print(f"Training Set", {X_train.shape})
print(f"Test Set", {X_test.shape})

#Step 5 Train the ML model
model = xgb.XGBRegressor(
    n_estimators = 100,
    max_depth = 1,                             # higher number here could cause overfitting
    learning_rate = 0.1                        # lower number here could cause stable learning
    )

model.fit(X_train, y_train)                    # model.fit() means "train the model" using training data
y_pred = model.predict(X_test)                 # X is capital because its a 2d matrix and y is small because its a 1D list (its a math convention)

#Step 6 Check the results of the predictions then go back to tuning the numbers or choose a different model if needed
r2 = r2_score(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred) ** 0.5)
mae = (mean_absolute_error(y_test, y_pred))
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} seconds")
print(f"MAE: {mae:.2f} seconds")


#Now tomorrow i have to learn how to make a result excel file so that i can see what was the correct answer and what it predicted.
#Also need to learning plotting so that i can see and understand underfitting and overfitting.