# After seeing low results of the linear regression model, i realised that my data is not linear and logarithmic
# 1 to 10 holes had a big jump in time but 100 to 110 holes had a small jump in time --- this is not linear
# The way I can avoid making this mistake again is to but clean the data and make scatter plots of the target column vs the features.
# The mistake I made was I jumped to linear regression before understanding the data. 
# When I make scatter plots I should ask my self
# Are the points in a straight line? --> Linear
# Are the points in a curve?         --> Log 
# Are the points scattered randomly? --> Maybe no relationship
# Is there acceleration?             --> Exponential

# Step 1 Import libraries/tools
import pandas as pd                # this is used to parse through excel files               
import matplotlib.pyplot as plt    # this is used for plotting and graphs
from sklearn.ensemble import GradientBoostingRegressor # this is the ML model, remember to not use classifier here
# The reason we are doing GB instead of XGboost is because I have less parameters to tune and its easier, XGboost is used when you want to squeeze out every last 0.1%
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error # this is the measurement of the results
from sklearn.preprocessing import StandardScaler

# Step 2 Load the data

df = pd.read_excel("7030 12kW.xlsx")

# Step 3 Clean the data

df = df.dropna()                # This will drop any row that has even a single missing value
df = df.drop_duplicates()       # This removes duplicates
# We dont need to use get dummies here because we dont have any categorical value to be converted into numeric

# Step 4 Exploratory Data Analysis to know what kind of data I have, is it linear or log

plt.figure(figsize=(12,4))

# Plot 1 is histogram of processing time

plt.subplot(1,2,1)                                                        # 1 rows, 2 columns, 1- on the left of the window
plt.hist(df['Processing_Time_Seconds'], bins=50, edgecolor ='black')
plt.xlabel("Processing Time Seconds")
plt.ylabel("Frequency")
plt.title("Distribution of processing time")

# Plot 2 is scatter - Piercing points vs processing time
plt.subplot(1,2,2)                                                        # 1 rows, 2 columns, 2- on the right of the window
plt.scatter(df['Piercing_Points'], df['Processing_Time_Seconds'], alpha=0.3, s=10)
plt.xlabel('Piercing Points (Holes)')
plt.ylabel('Processing Time (Seconds)')
plt.title('Piercing Points vs Processing Time\n(Is this LINEAR or CURVED?)')

plt.tight_layout()    # This makes sure nothing is overlapping and everything is spaced cleanly

#plt.show()

# Step 5 Split the data to train and assign target column 

X = df.drop('Processing_Time_Seconds', axis = 1) # This is used to drop the column mentioned and everything else becomes the features column
y = df['Processing_Time_Seconds']  # This sets the target column

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Step 6 Train the model

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train) # Model guesses on training data
y_test_pred = model.predict(X_test)   # Model guesses on test data


# Step 7 see the metrics to know how accurate are the predictions

r2_train = r2_score(y_train, y_train_pred)
r2_test= r2_score(y_test, y_test_pred)

print(f"r2_train:{r2_train}")
print(f"r2_test: {r2_test}")

# Calculate R² for both train and test
r2_train = r2_score(y_train, y_train_pred)      # ✅ How good on training data
r2_test = r2_score(y_test, y_test_pred)         # ✅ How good on new data

# Calculate MAE (Mean Absolute Error) - average error in seconds
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate MAPE (Mean Absolute Percentage Error) - error as a percentage
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

# Print results nicely
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

print("\n📊 TRAINING DATA (Model learned from this):")
print(f"   R² Score:  {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"   MAE:       {mae_train:.2f} seconds (average error)")
print(f"   MAPE:      {mape_train:.4f} ({mape_train*100:.2f}%)")

print("\n📊 TESTING DATA (Never seen this before):")
print(f"   R² Score:  {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"   MAE:       {mae_test:.2f} seconds (average error)")
print(f"   MAPE:      {mape_test:.4f} ({mape_test*100:.2f}%)")

print("\n⚠️  OVERFITTING CHECK:")
r2_gap = r2_train - r2_test
print(f"   Train R² - Test R² = {r2_gap:.4f} ({r2_gap*100:.2f}%)")

if r2_gap < 0.05:
    print(f"   ✅ EXCELLENT: Minimal overfitting")
elif r2_gap < 0.10:
    print(f"   ✅ GOOD: Some overfitting, but acceptable")
else:
    print(f"   ⚠️  WARNING: Significant overfitting detected")

print("\n" + "="*70)

# Step 8: Visualize the curved relationship --- [PASTED]

import numpy as np

# Sort by Piercing_Points so the line is smooth
sorted_idx = np.argsort(X_test['Piercing_Points'])
X_sorted = X_test.iloc[sorted_idx]
y_sorted = y_test.iloc[sorted_idx]
y_pred_sorted = y_test_pred[sorted_idx]

plt.figure(figsize=(12, 5))

# Plot 1: Raw curved data
plt.subplot(1, 2, 1)
plt.scatter(df['Piercing_Points'], df['Processing_Time_Seconds'], 
            alpha=0.3, s=10, label='Actual Data')
plt.xlabel('Piercing Points (Holes)')
plt.ylabel('Processing Time (Seconds)')
plt.title('Raw Data: CURVED (Logarithmic)')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Model predictions follow the curve
plt.subplot(1, 2, 2)
plt.scatter(X_sorted['Piercing_Points'], y_sorted, 
            alpha=0.5, s=20, label='Actual', color='blue')
plt.plot(X_sorted['Piercing_Points'], y_pred_sorted, 
         'r-', linewidth=2, label='Model Prediction')
plt.xlabel('Piercing Points (Holes)')
plt.ylabel('Processing Time (Seconds)')
plt.title('Model Learns the Curved Pattern')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Step: Check which features are important
model_scaled = GradientBoostingRegressor(random_state=42)
model_scaled.fit(X, y)

# Get feature importance
feature_importance = model_scaled.feature_importances_

# Create a sorted list
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df)
print("\n--- Top 10 Most Important Features ---")
print(importance_df.head(10))



# Step  we export results in excel to compare.
result = pd.DataFrame({                                             # Creates a table with columns inside the { }
    'Actual Processing Time': y_test.values,                        # first column in the table
    'Predicted Processing Time': y_test_pred,                            # Second column in the table
    'Error': y_test.values - y_test_pred                                 # Third column
})

result.to_excel('Result.xlsx', index = False)
print("Printed")

