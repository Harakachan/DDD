import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng as rng
CSV_FILE = "sustainable_waste_management_dataset_2024.csv"
df = pd.read_csv(CSV_FILE)
df.head()

selected_features = ['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'is_weekend', 'is_holiday', 'recycling_campaign', 'temp_c', 'rain_mm']
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

st.write("R squared: ", r2_score(Y_test, Y_pred))

plt.figure(figsize=(10, 6))
st.pyplot(figsize=(10, 6))

plt.scatter(Y_test, Y_pred, alpha=0.7)
st.scatter_chart(Y_test, Y_pred, alpha=0.7)

plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='pink', lw=2, label='Perfect Prediction Line')
st.pyplot([y.min(), y.max()], [y.min(), y.max()], '--', color='pink', lw=2, label='Perfect Prediction Line')

plt.legend()
plt.grid(True)
plt.show()
st.scatter_chart(fig)