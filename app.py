import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="소비자 행동 분석 대시보드", layout="wide")
st.title("\U0001F6CD️ 고급 소비자 행동 분석 대시보드")

@st.cache_data
def load_data():
    df = pd.read_csv("shopping_behavior_updated.csv")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60, 100],
                            labels=["18-24", "25-34", "35-44", "45-59", "60+"])
    return df

df = load_data()

st.sidebar.header("\U0001F50D 필터")
gender = st.sidebar.multiselect("성별 선택", options=df["Gender"].unique(), default=df["Gender"].unique())
channel = st.sidebar.multiselect("배송 방식 (Shipping Type)", options=df["Shipping Type"].unique(), default=df["Shipping Type"].unique())
payment = st.sidebar.multiselect("결제 수단", options=df["Payment Method"].unique(), default=df["Payment Method"].unique())

filtered_df = df[
    (df["Gender"].isin(gender)) &
    (df["Shipping Type"].isin(channel)) &
    (df["Payment Method"].isin(payment))
]

st.sidebar.markdown(f"\U0001F3AF 총 {len(filtered_df)}명 선택됨")

st.write("현재 별점 데이터 개수:", filtered_df['Review Rating'].notnull().sum())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001F4CA 개요 시각화", "\U0001F4C8 고급 분석", "\U0001F52C 상관관계",
    "\U0001F916 별점 예측", "\U0001F381 추천 시스템", "\u2B50 별점 행동 분석"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 성별 구매 비율")
        fig, ax = plt.subplots()
        gender_counts = filtered_df["Gender"].value_counts()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    with col2:
        st.subheader("2. 배송 방식 비율")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x="Shipping Type", palette="pastel", ax=ax)
        st.pyplot(fig)
    st.subheader("3. 연령대별 상품 선호 (Stacked Bar)")
    age_cat = filtered_df.groupby(['AgeGroup', 'Category']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    age_cat.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("4. 결제 수단별 평균 지출")
        avg_spend = filtered_df.groupby("Payment Method")["Purchase Amount (USD)"].mean().sort_values(ascending=False)
        st.bar_chart(avg_spend)
    with col2:
        st.subheader("5. 연령대별 평균 지출")
        age_spend = filtered_df.groupby("AgeGroup")["Purchase Amount (USD)"].mean()
        fig = px.line(x=age_spend.index, y=age_spend.values, markers=True, labels={"x":"Age Group", "y":"Avg Spend"})
        st.plotly_chart(fig)
    st.subheader("6. 상품 카테고리별 지출 Boxplot")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=filtered_df, x="Categ
