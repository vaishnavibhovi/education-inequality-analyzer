import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Education Inequality Analyzer", layout="wide")

st.title("📊 Education Inequality Analyzer (India)")
st.markdown("Analyzing literacy, gender gap, and dropout trends")

df = pd.read_csv("education_data.csv")

st.sidebar.header("Filters")
selected_state = st.sidebar.multiselect("Select States", df["State"], default=df["State"])
filtered_df = df[df["State"].isin(selected_state)]

st.subheader("📂 Dataset")
st.dataframe(filtered_df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Literacy Rate")
    fig, ax = plt.subplots()
    ax.bar(filtered_df["State"], filtered_df["Literacy_Rate"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

filtered_df["Gender_Gap"] = filtered_df["Male_Literacy"] - filtered_df["Female_Literacy"]

with col2:
    st.subheader("⚖️ Gender Gap")
    fig2, ax2 = plt.subplots()
    ax2.bar(filtered_df["State"], filtered_df["Gender_Gap"])
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    X = df[["Literacy_Rate", "Male_Literacy", "Female_Literacy", "Rural_Population"]]
y = df["Dropout_Rate"]

model = LinearRegression()
model.fit(X, y)

st.subheader("🤖 Predict Dropout Rate")

lit = st.slider("Literacy Rate", 50, 100, 70)
male = st.slider("Male Literacy", 50, 100, 75)
female = st.slider("Female Literacy", 50, 100, 65)
rural = st.slider("Rural Population %", 30, 100, 60)

prediction = model.predict([[lit, male, female, rural]])

st.success(f"Predicted Dropout Rate: {prediction[0]:.2f}%")

st.subheader("🌍 Key Insights")
st.write("""
- Higher rural population correlates with higher dropout rates  
- Gender gap significantly impacts education access  
""")

st.subheader("💡 Policy Recommendations")
st.write("""
- Improve rural school infrastructure  
- Promote girls’ education programs  
""")