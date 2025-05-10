import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# 1) Load data
file_path = '50_Startups.csv'
df = pd.read_csv(file_path)

# 2) Check for missing values
assert not df.isnull().values.any(), "Missing values detected in the dataset"

# 3) Basic statistical description to check for outliers
print("\nStatistical Description:")
print(df.describe())

# 4) Encode categorical column 'State' using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)

# 5) Separate features and target variable
X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

# 6) Standardize numeric features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 7) Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8) Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9) Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"\nR² = {r2:.3f}")
print(f"RMSE = {rmse:,.2f} USD")

# 10) Display coefficients
coef_df = pd.Series(model.coef_, index=X.columns)
print("\nCoefficients ranked by impact:")
print(coef_df.reindex(coef_df.abs().sort_values(ascending=False).index))

# 11) Statistical analysis using statsmodels
X_const = sm.add_constant(X_scaled)
ols = sm.OLS(y, X_const).fit()
print("\nOLS Model Summary:")
print(ols.summary())

# 12) Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
print("\nVIF Analysis (values < 5 are acceptable):")
print(vif_data.sort_values('VIF', ascending=False))

# 13) Identify the company with the highest actual profit
max_profit_row = df.loc[df['Profit'].idxmax()]
print("\nCompany with Highest Actual Profit:")
print(max_profit_row)
