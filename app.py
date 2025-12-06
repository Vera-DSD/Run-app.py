import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🏠 House Price Predictor")

# Загружаем модели
try:
    model = joblib.load('GB_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("✅ Модели загружены")
except:
    st.error("❌ Ошибка загрузки файлов")
    st.stop()

# Загрузка CSV
uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"📊 Загружено: {len(df)} записей")
    
    if st.button("🎯 Сделать предсказания", type="primary"):
        with st.spinner("Обрабатываю данные..."):
            try:
                # Применяем препроцессор
                X_transformed = preprocessor.transform(df)
                
                # Преобразуем в numpy array
                if isinstance(X_transformed, pd.DataFrame):
                    X_array = X_transformed.values
                else:
                    X_array = X_transformed
                
                # Делаем предсказания
                predictions = model.predict(X_array)
                
                # Результаты
                results = pd.DataFrame({
                    'Id': df['Id'] if 'Id' in df.columns else range(1, len(df)+1),
                    'SalePrice': predictions
                })
                
                # Показываем
                st.success("✅ Готово!")
                st.dataframe(results.head(20))
                
                # Статистика
                st.write(f"**Средняя цена:** ${predictions.mean():,.0f}")
                st.write(f"**Диапазон:** ${predictions.min():,.0f} - ${predictions.max():,.0f}")
                
                # Скачать
                csv = results.to_csv(index=False)
                st.download_button("📥 Скачать результаты", csv, "predictions.csv")
                
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")