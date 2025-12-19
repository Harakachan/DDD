import streamlit as st

# ================== PAGE CONFIG (ต้องอยู่บนสุด) ==================
st.set_page_config(
    page_title="Waste Intelligence Dashboard",
    page_icon="♻️",
    layout="wide"
)

# ================== IMPORT ==================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ================== CSS ==================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #2ECC71;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 30px;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ================== TITLE ==================
st.markdown('<div class="big-title">♻️ Waste Intelligence System</div>', unsafe_allow_html=True)
st.write("Linear Regression + Waste Analytics Dashboard")

# ================== LOAD DATA ==================
CSV_FILE = "sustainable_waste_management_dataset_2024.csv"
df = pd.read_csv(CSV_FILE)

# ================== MODEL ==================
features = [
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

df_model = df[features + ['waste_kg']].dropna()

X = df_model[features]
y = df_model['waste_kg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ================== METRICS ==================
st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
with c2:
    st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):,.2f}")

# ================== SCATTER PLOT ==================
st.markdown('<div class="section-title">🔵 Predicted vs Actual Waste</div>', unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(y_test, y_pred, s=40, alpha=0.8)
min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
ax1.plot([min_v, max_v], [min_v, max_v], '--', lw=2)
ax1.set_xlabel("Actual Waste (kg)")
ax1.set_ylabel("Predicted Waste (kg)")
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)

# ================== BAR CHART: DAY OF WEEK ==================
st.markdown('<div class="section-title">📅 Average Waste by Day of Week</div>', unsafe_allow_html=True)

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()

order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

avg_waste = (
    df.groupby("day_of_week")["waste_kg"]
      .mean()
      .reindex(order)
)

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar(avg_waste.index, avg_waste.values)
ax2.set_xlabel("Day of Week")
ax2.set_ylabel("Average Waste (kg)")
ax2.set_title("Average Waste per Day (Mon–Sun)")
ax2.grid(axis="y", alpha=0.3)

st.pyplot(fig2)

# ================== FOOTER ==================
st.markdown("---")
st.caption("Built for competition-grade analytics • Linear Regression • Streamlit")
