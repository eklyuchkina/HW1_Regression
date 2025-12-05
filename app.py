import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    with open("models/ridge_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.title("Предсказание стоимости автомобиля")

tab_eda, tab_predict, tab_weights = st.tabs(["EDA", "Предсказание", "Веса модели"])

with tab_eda:
    st.header("Исследовательский анализ данных")
    uploaded = st.file_uploader("Загрузить CSV для анализа", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
        st.dataframe(df.head())
        plot_type = st.selectbox("Выберите график", ["Гистограмма целевой переменной", "Корреляция числовых признаков"])
        if plot_type == "Гистограмма целевой переменной":
            target_col = st.selectbox("Столбец целевой переменной", df.select_dtypes(include=[np.number]).columns)
            fig, ax = plt.subplots()
            sns.histplot(df[target_col], bins=40, kde=True, ax=ax)
            st.pyplot(fig)
        if plot_type == "Корреляция числовых признаков":
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

with tab_predict:
    st.header("Предсказание стоимости")
    file = st.file_uploader("Загрузить CSV с признаками объектов", type=["csv"])
    if file:
        df_new = pd.read_csv(file)
        st.subheader("Входные данные")
        st.dataframe(df_new.head())

        if "selling_price" in df_new.columns:
            df_new = df_new.drop(columns=["selling_price"])

        missing_cols = set(model.feature_names_in_) - set(df_new.columns)
        extra_cols = set(df_new.columns) - set(model.feature_names_in_)

        if missing_cols:
            st.error(f"В данных отсутствуют признаки: {sorted(missing_cols)}")
        else:
            X = df_new[model.feature_names_in_]
            preds = model.predict(X)
            result = df_new.copy()
            result["predicted_price"] = preds
            st.subheader("Результаты предсказаний")
            st.dataframe(result.head())
            csv_out = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )

with tab_weights:
    st.header("Веса модели Ridge")
    coefs = pd.DataFrame({
        "feature": model.feature_names_in_,
        "coef": model.coef_
    })
    coefs = coefs.sort_values("coef", ascending=False)
    st.dataframe(coefs)

    top_n = st.slider("Количество топ признаков", 5, len(coefs), 10)
    top = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="coef", y="feature", data=top, ax=ax)
    st.pyplot(fig)