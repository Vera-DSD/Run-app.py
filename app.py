iimport streamlit as st
import pandas as pd
import joblib
import numpy as np

# Простая конфигурация без кэширования
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.write("Загрузите CSV файл для предсказания цен на дома")

# Загрузка модели и препроцессора
try:
    model = joblib.load('GB_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("✅ Модель и препроцессор загружены")
except Exception as e:
    st.error(f"❌ Ошибка загрузки: {e}")
    st.stop()

# Простой интерфейс с одной вкладкой
uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])

if uploaded_file is not None:
    try:
        # Чтение файла
        df = pd.read_csv(uploaded_file)
        
        st.write(f"**Загружено:** {df.shape[0]} строк, {df.shape[1]} колонок")
        
        # Показать превью
        if st.checkbox("Показать данные"):
            st.dataframe(df.head())
        
        # Кнопка для предсказания
        if st.button("🎯 Сделать предсказания", type="primary"):
            with st.spinner("Обрабатываю данные..."):
                try:
                    # Применяем препроцессор
                    X_processed = preprocessor.transform(df)
                    
                    # Делаем предсказания
                    predictions = model.predict(X_processed)
                    
                    # Создаем результаты
                    if 'Id' in df.columns:
                        results = pd.DataFrame({
                            'Id': df['Id'],
                            'SalePrice': predictions
                        })
                    else:
                        results = pd.DataFrame({
                            'Id': range(1, len(df) + 1),
                            'SalePrice': predictions
                        })
                    
                    st.success("✅ Предсказания готовы!")
                    
                    # Статистика
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Средняя цена", f"${predictions.mean():,.0f}")
                    with col2:
                        st.metric("Минимальная цена", f"${predictions.min():,.0f}")
                    with col3:
                        st.metric("Максимальная цена", f"${predictions.max():,.0f}")
                    
                    # Показать таблицу
                    st.write("**Первые 10 результатов:**")
                    st.dataframe(results.head(10))
                    
                    # Скачать все результаты
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Скачать все результаты (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)[:200]}")
                    
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")

# Простая форма для тестирования
st.markdown("---")
st.write("### Тест с примером данных")

# Создаем пример данных
example_data = {
    'MSSubClass': 60,
    'MSZoning': 'RL',
    'LotFrontage': 65.0,
    'LotArea': 8450,
    'Street': 'Pave',
    'Alley': 'NA',
    'LotShape': 'Reg',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Inside',
    'LandSlope': 'Gtl',
    'Neighborhood': 'NAmes',
    'Condition1': 'Norm',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '1Story',
    'OverallQual': 7,
    'OverallCond': 5,
    'YearBuilt': 2003,
    'YearRemodAdd': 2003,
    'RoofStyle': 'Gable',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'MasVnrType': 'BrkFace',
    'MasVnrArea': 196.0,
    'ExterQual': 'Gd',
    'ExterCond': 'TA',
    'Foundation': 'PConc',
    'BsmtQual': 'Gd',
    'BsmtCond': 'TA',
    'BsmtExposure': 'No',
    'BsmtFinType1': 'GLQ',
    'BsmtFinSF1': 706,
    'BsmtFinType2': 'Unf',
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 150,
    'TotalBsmtSF': 856,
    'Heating': 'GasA',
    'HeatingQC': 'Ex',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    '1stFlrSF': 856,
    '2ndFlrSF': 854,
    'LowQualFinSF': 0,
    'GrLivArea': 1710,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'FullBath': 2,
    'HalfBath': 1,
    'BedroomAbvGr': 3,
    'KitchenAbvGr': 1,
    'KitchenQual': 'Gd',
    'TotRmsAbvGrd': 8,
    'Functional': 'Typ',
    'Fireplaces': 0,
    'FireplaceQu': 'NA',
    'GarageType': 'Attchd',
    'GarageYrBlt': 2003.0,
    'GarageFinish': 'RFn',
    'GarageCars': 2,
    'GarageArea': 548,
    'GarageQual': 'TA',
    'GarageCond': 'TA',
    'PavedDrive': 'Y',
    'WoodDeckSF': 0,
    'OpenPorchSF': 61,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    'PoolQC': 'NA',
    'Fence': 'NA',
    'MiscFeature': 'NA',
    'MiscVal': 0,
    'MoSold': 2,
    'YrSold': 2008,
    'SaleType': 'WD',
    'SaleCondition': 'Normal'
}

if st.button("🚀 Тест на примере данных"):
    with st.spinner("Выполняю тестовое предсказание..."):
        try:
            # Создаем DataFrame
            df_test = pd.DataFrame([example_data])
            
            # Применяем препроцессор
            X_processed = preprocessor.transform(df_test)
            
            # Делаем предсказание
            prediction = model.predict(X_processed)[0]
            
            st.success(f"🏡 Тестовое предсказание: **${prediction:,.0f}**")
            
        except Exception as e:
            st.error(f"❌ Тестовая ошибка: {e}")

# Инструкция
st.markdown("---")
st.write("""
### 📋 Инструкция:
1. Загрузите CSV файл с данными о домах
2. Нажмите кнопку "Сделать предсказания"
3. Скачайте результаты в CSV формате

### ⚠️ Требования к файлу:
- Должен содержать все 79 признаков
- Формат как в train.csv Kaggle
""")