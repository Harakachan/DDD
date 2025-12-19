import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================
# Title
# =====================
st.title("Waste Prediction using Linear Regression")


df = pd.read_csv("sustainable_waste_management_dataset_2024.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =====================
# Feature Selection
# =====================
selected_features = [
    'population',
    'recyclable_kg',
    'organic_kg',
    'collection_capacity_kg',
    'overflow',
    'is_weekend',
    'is_holiday',
    'recycling_campaign',
    'temp_c',
    'rain_mm'
]

X = df[selected_features]
y = df['waste_kg']

# =====================
# Handle Missing Values
# =====================
df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

st.write("Rows after dropna:", df_combined.shape[0])

# =====================
# Train/Test Split
# =====================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Train Model
# =====================
model = LinearRegression()
model.fit(X_train, Y_train)

# =====================
# Prediction
# =====================
Y_pred = model.predict(X_test)

# =====================
# Metrics (แทน print)
# =====================
st.subheader("Model Performance")

r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

st.write("R squared:", r2)
st.write("Mean Squared Error:", mse)

# =====================
# Plot (เหมือน Colab)
# =====================
st.subheader("Predicted vs Actual Waste")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    Y_test.values,
    Y_pred,
    alpha=0.7
)

ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    '--',
    color='pink',
    lw=2,
    label='Perfect Prediction Line'
)

ax.set_xlabel("Actual waste")
ax.set_ylabel("Predicted waste")
ax.set_title("Predicted vs Actual waste")
ax.legend()
ax.grid(True)

st.pyplot(fig)
