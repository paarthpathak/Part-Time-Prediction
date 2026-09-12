"""
Part Processing Time Predictor - Gradient Boosting Model

# After seeing low results of the linear regression model, I realised that my data is not linear and logarithmic

Discovered that processing time relationship is logarithmic (not linear):
- 1-10 holes: big time jump
- 100-110 holes: small time jump
--- this is not linear

Key insight: Always visualize data before choosing algorithm.
"""


import pandas as pd                             
import matplotlib.pyplot as plt   
from sklearn.ensemble import GradientBoostingRegressor 
# The reason we are doing GB instead of XGboost is because I have less parameters to tune and its easier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error 
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("data.xlsx")



df = df.dropna()               
df = df.drop_duplicates()      



plt.figure(figsize=(12,4))



plt.subplot(1,2,1)                                                        
plt.hist(df['Processing_Time_Seconds'], bins=50, edgecolor ='black')
plt.xlabel("Processing Time Seconds")
plt.ylabel("Frequency")
plt.title("Distribution of processing time")

plt.subplot(1,2,2)                                                        
plt.scatter(df['Piercing_Points'], df['Processing_Time_Seconds'], alpha=0.3, s=10)
plt.xlabel('Piercing Points (Holes)')
plt.ylabel('Processing Time (Seconds)')
plt.title('Piercing Points vs Processing Time\n(Is this LINEAR or CURVED?)')

plt.tight_layout()    

X = df.drop('Processing_Time_Seconds', axis = 1) 
y = df['Processing_Time_Seconds']  

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train) 
y_test_pred = model.predict(X_test)   



r2_train = r2_score(y_train, y_train_pred)
r2_test= r2_score(y_test, y_test_pred)

print(f"r2_train:{r2_train}")
print(f"r2_test: {r2_test}")

r2_train = r2_score(y_train, y_train_pred)      
r2_test = r2_score(y_test, y_test_pred)         

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)


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
plt.savefig('Gradient_Boosting_Plot.png', dpi=300, bbox_inches='tight')
# plt.show()

# Step: Check which features are important
model_scaled = GradientBoostingRegressor(random_state=42)
model_scaled.fit(X, y)

# Get feature importance
feature_importance = model_scaled.feature_importances_


importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df)
print("\n--- Top 10 Most Important Features ---")
print(importance_df.head(10))


result = pd.DataFrame({                                             
    'Actual Processing Time': y_test.values,                        
    'Predicted Processing Time': y_test_pred,                            
    'Error': y_test.values - y_test_pred                                 
})

result.to_excel('Result.xlsx', index = False)
print("Printed")

