# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("Salary_Data.csv")

# Train the model
X = data[['YearsExperience']]
y = data['Salary']
model = LinearRegression()
model.fit(X, y)

# Title and intro
st.title("💼 Salary Predictor App")
st.write("Enter years of experience to predict the expected salary.")

# User input
years_exp = st.slider("👨‍💻 Years of Experience", 0.0, 20.0, 2.0)

# Prediction
predicted_salary = model.predict([[years_exp]])[0]
st.subheader("💰 Predicted Salary:")
st.success(f"₹{predicted_salary:,.2f}")

# Plot (optional but cool)
st.subheader("📈 Experience vs Salary")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X), color='red', label='Regression Line')
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.legend()
st.pyplot(fig)
