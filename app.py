import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="소비자 행동 분석 대시보드", layout="wide")
st.title("🛍️ 고급 소비자 행동 분석 대시보드")

# ----------------------
# Load Data
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("shopping_behavior_updated.csv")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60, 100],
                            labels=["18-24", "25-34", "35-44", "45-59", "60+"])
    return df

df = load_data()

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.header("🔍 필터")
gender = st.sidebar.multiselect("성별 선택", options=df["Gender"].unique(), default=df["Gender"].unique())
channel = st.sidebar.multiselect("배송 방식 (Shipping Type)", options=df["Shipping Type"].unique(), default=df["Shipping Type"].unique())
payment = st.sidebar.multiselect("결제 수단", options=df["Payment Method"].unique(), default=df["Payment Method"].unique())

filtered_df = df[
    (df["Gender"].isin(gender)) &
    (df["Shipping Type"].isin(channel)) &
    (df["Payment Method"].isin(payment))
]

st.sidebar.markdown(f"🎯 총 {len(filtered_df)}명 선택됨")

# ----------------------
# Tabs for Layout
# ----------------------
tab1, tab2, tab3 = st.tabs(["📊 개요 시각화", "📈 고급 분석", "🔬 상관관계"])

# ----------------------
# TAB 1: Overview Visuals
# ----------------------
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

# ----------------------
# TAB 2: 고급 분석
# ----------------------
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
    sns.boxplot(data=filtered_df, x="Category", y="Purchase Amount (USD)", palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

# ----------------------
# TAB 3: Correlation
# ----------------------
with tab3:
    st.subheader("7. 수치형 변수 상관관계 히트맵")
    num_cols = filtered_df.select_dtypes(include='number')
    corr = num_cols.corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.caption("🔍 예: 'Purchase Amount'와 나이 또는 구매 빈도 간의 상관관계 등 확인 가능")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with st.expander("🤖 머신러닝: 별점 예측"):
    st.markdown("이 분석은 고객 정보와 구매 특성으로부터 별점을 예측합니다.")
    
    ml_df = filtered_df.copy()
    ml_df = ml_df.dropna(subset=['Review Rating'])  # 별점 없는 경우 제거

    # 숫자로 변환
    ml_df_encoded = pd.get_dummies(ml_df[["Age", "Gender", "Category", "Payment Method", "Shopping Channel"]])
    ml_df_encoded["Purchase"] = ml_df["Purchase Amount (USD)"]
    X = ml_df_encoded
    y = ml_df["Review Rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.write(f"예측 RMSE (낮을수록 정확): {rmse:.2f}")

    # 사용자 입력으로 예측 테스트
    st.markdown("### 🎯 별점 예측 테스트")
    test_input = {
        "Age": st.slider("나이", 18, 70, 30),
        "Purchase": st.slider("구매금액", 10, 1000, 100),
        "Gender_Female": st.radio("성별", ["Female", "Male"]) == "Female",
        "Category_Clothing": st.radio("카테고리", ["Clothing", "Accessories", "Shoes"]) == "Clothing",
        "Payment Method_Credit Card": st.radio("결제 방식", ["Credit Card", "Paypal", "Cash"]) == "Credit Card",
        "Shopping Channel_Online": st.radio("쇼핑 채널", ["Online", "Offline"]) == "Online"
    }

    test_df = pd.DataFrame([test_input])
    y_test_pred = model.predict(test_df)[0]
    st.success(f"예측 별점: {y_test_pred:.2f} / 5.0")

with st.expander("🎁 추천 시스템: 자주 사는 항목 기반 추천"):
    st.markdown("최근 자주 구매한 카테고리/상품을 기반으로 다른 유사한 아이템을 추천합니다.")

    # 가장 많이 산 카테고리 찾기
    most_bought = filtered_df["Category"].mode().iloc[0]
    st.write(f"🛍️ 이 고객군이 가장 많이 산 카테고리: `{most_bought}`")

    # 해당 카테고리를 산 사람들의 다른 상품 추천
    similar_users = df[df["Category"] == most_bought]
    recommended_items = similar_users["Item Purchased"].value_counts().head(5)

    st.write("📦 추천 상품:")
    for item, count in recommended_items.items():
        st.markdown(f"- {item} ({count}명 구매)")
with st.expander("⭐ 리뷰 별점에 따른 행동 분석"):
    st.markdown("별점(Review Rating) 분포와 관련된 행동 특성을 분석합니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 별점 분포")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["Review Rating"].dropna(), bins=5, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("2. 결제 수단에 따른 별점 평균")
        rating_by_payment = filtered_df.groupby("Payment Method")["Review Rating"].mean()
        st.bar_chart(rating_by_payment)

    st.subheader("3. 카테고리별 평균 별점")
    rating_by_cat = filtered_df.groupby("Category")["Review Rating"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=rating_by_cat.index, y=rating_by_cat.values, palette="Blues_d", ax=ax)
    ax.set_ylabel("평균 별점")
    st.pyplot(fig)
