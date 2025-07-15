import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🛍️ 소비자 행동 분석 대시보드")

# CSV 파일 불러오기
df = pd.read_csv("shopping_behavior_updated.csv")

# 성별 구매 비율
st.subheader("1. 성별 구매 비율")
gender_count = df['Gender'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90)
st.pyplot(fig1)

# 연령대별 선호 카테고리
st.subheader("2. 연령대별 상품 선호")
age_bins = [18, 25, 35, 45, 60]
age_labels = ['18-24', '25-34', '35-44', '45-59']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
grouped = df.groupby(['AgeGroup', 'Category']).size().unstack().fillna(0)

fig2, ax2 = plt.subplots(figsize=(10, 4))
grouped.plot(kind='bar', stacked=True, ax=ax2)
plt.xticks(rotation=0)
st.pyplot(fig2)
