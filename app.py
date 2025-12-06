import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.markdown("### Предсказание цен на дома с использованием Gradient Boosting")

def transform_new_data(X_new, transformers):
    numerical_features = transformers['numerical_features']
    categorical_features = transformers['categorical_features']
    numeric_imputer = transformers['numeric_imputer']
    scaler = transformers['scaler']
    cat_imputer = transformers['cat_imputer']
    label_encoders = transformers['label_encoders']
    
    # Ensure all required columns exist
    for col in numerical_features:
        if col not in X_new.columns:
            X_new[col] = 0
    for col in categorical_features:
        if col not in X_new.columns:
            X_new[col] = 'NA'
    
    # Numerical
    X_num = numeric_imputer.transform(X_new[numerical_features])
    X_num = scaler.transform(X_num)
    
    # Categorical
    X_cat = cat_imputer.transform(X_new[categorical_features])
    X_cat_encoded = np.zeros(X_cat.shape, dtype=np.float64)
    
    for i, col in enumerate(categorical_features):
        le = label_encoders[col]
        for j, val in enumerate(X_cat[:, i]):
            if val in le.classes_:
                X_cat_encoded[j, i] = le.transform([val])[0]
            else:
                X_cat_encoded[j, i] = -1
    
    return np.hstack([X_num, X_cat_encoded])

@st.cache_resource
def load_models():
    try:
        model = joblib.load('GBB_model.pkl')
        transformers = joblib.load('transformers2.pkl')
        feature_info = joblib.load('feature2_info.pkl')
        st.success(f"✅ Модель загружена (использует {transformers['total_features']} признаков)")
        return model, transformers, feature_info
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return None, None, None

model, transformers, feature_info = load_models()

if model and transformers and feature_info:
    with st.sidebar:
        st.header("ℹ️ Информация о модели")
        st.write(f"**Тип:** GradientBoostingRegressor")
        st.write(f"**Признаков:** {transformers['total_features']}")
        st.write(f"**Деревьев:** {model.n_estimators}")
        
        st.header("📊 Используемые признаки")
        with st.expander("Числовые признаки"):
            for feat in feature_info['numerical_features']:
                st.write(f"- {feat}")
        with st.expander("Категориальные признаки"):
            for feat in feature_info['categorical_features']:
                st.write(f"- {feat}")
    
    tab1, tab2 = st.tabs(["📤 Загрузка CSV", "📝 Ручной ввод"])
    
    with tab1:
        st.header("Загрузите CSV файл")
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Загружено: {df.shape[0]} строк")
                if st.checkbox("Показать данные"):
                    st.dataframe(df.head())
                if st.button("🎯 Сделать предсказания", type="primary"):
                    with st.spinner("Обрабатываю..."):
                        X_processed = transform_new_data(df, transformers)
                        predictions = model.predict(X_processed)
                        results = pd.DataFrame({
                            'Id': df['Id'] if 'Id' in df.columns else range(1, len(df) + 1),
                            'PredictedPrice': predictions
                        })
                        st.success("✅ Предсказания готовы!")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Средняя", f"${predictions.mean():,.0f}")
                        col2.metric("Медиана", f"${np.median(predictions):,.0f}")
                        col3.metric("Диапазон", f"${predictions.min():,.0f}–${predictions.max():,.0f}")
                        st.dataframe(results.head(20))
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Скачать результаты",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    
    with tab2:
        st.header("Ручной ввод")
        with st.form("input_form"):
            st.subheader("Основные параметры")
            col1, col2 = st.columns(2)
            with col1:
                overall_qual = st.slider("Общее качество", 1, 10, 7)
                gr_liv_area = st.number_input("Жилая площадь", 500, 5000, 1500)
                total_bsmt_sf = st.number_input("Площадь подвала", 0, 3000, 1000)
                year_built = st.number_input("Год постройки", 1900, 2024, 2000)
            with col2:
                lot_area = st.number_input("Площадь участка", 1000, 50000, 10000)
                garage_cars = st.slider("Машиномест", 0, 4, 2)
                full_bath = st.slider("Полных ванных", 0, 4, 2)
                fireplaces = st.slider("Камины", 0, 4, 1)
            
            with st.expander("Дополнительно"):
                mszoning = st.selectbox("Зонирование", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
                neighborhood = st.selectbox("Район", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'])
                kitchen_qual = st.selectbox("Качество кухни", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            
            submitted = st.form_submit_button("💰 Предсказать цену")
            if submitted:
                with st.spinner("Рассчитываю..."):
                    try:
                        data = {}
                        # Numerical
                        for col in feature_info['numerical_features']:
                            if col == 'OverallQual':
                                data[col] = overall_qual
                            elif col == 'GrLivArea':
                                data[col] = gr_liv_area
                            elif col == 'TotalBsmtSF':
                                data[col] = total_bsmt_sf
                            elif col == 'YearBuilt':
                                data[col] = year_built
                            elif col == 'LotArea':
                                data[col] = lot_area
                            elif col == 'GarageCars':
                                data[col] = garage_cars
                            elif col == 'FullBath':
                                data[col] = full_bath
                            elif col == 'Fireplaces':
                                data[col] = fireplaces
                            elif 'Year' in col:
                                data[col] = year_built
                            elif col == 'OverallCond':
                                data[col] = 5
                            else:
                                data[col] = 0  # default for others
                        
                        # Categorical
                        for col in feature_info['categorical_features']:
                            if col == 'MSZoning':
                                data[col] = mszoning
                            elif col == 'Neighborhood':
                                data[col] = neighborhood
                            elif col == 'KitchenQual':
                                data[col] = kitchen_qual
                            elif col == 'CentralAir':
                                data[col] = 'Y'
                            elif col == 'PavedDrive':
                                data[col] = 'Y'
                            elif col == 'SaleCondition':
                                data[col] = 'Normal'
                            else:
                                data[col] = 'NA'
                        
                        df_input = pd.DataFrame([data])
                        X_processed = transform_new_data(df_input, transformers)
                        prediction = model.predict(X_processed)[0]
                        st.success(f"## 🏡 Предсказанная цена: **${prediction:,.0f}**")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)[:200]}")
else:
    st.warning("⚠️ Сначала обучите модель: `python train_model.py`")