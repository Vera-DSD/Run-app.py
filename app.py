import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Настройка страницы
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Заголовок
st.title("🏠 House Price Predictor")
st.markdown("### Предсказание цен на дома с использованием Gradient Boosting")

# Загрузка модели и препроцессора
@st.cache_resource
def load_model():
    try:
        model = joblib.load('GB_model.pkl')
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        return preprocessor
    except Exception as e:
        st.error(f"❌ Ошибка загрузки препроцессора: {e}")
        return None

# Загружаем
model = load_model()
preprocessor = load_preprocessor()

if model and preprocessor:
    st.success("✅ Модель и препроцессор успешно загружены!")
    
    # Показать информацию о модели
    with st.expander("ℹ️ Информация о модели"):
        st.write(f"**Тип модели:** {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            st.write(f"**Количество деревьев:** {model.n_estimators}")
        if hasattr(model, 'feature_names_in_'):
            st.write(f"**Используется признаков:** {len(model.feature_names_in_)}")
    
    # Основной интерфейс
    tab1, tab2 = st.tabs(["📤 Загрузка CSV", "📝 Ручной ввод"])
    
    with tab1:
        st.header("Загрузите CSV файл с данными")
        
        uploaded_file = st.file_uploader(
            "Выберите CSV файл", 
            type=['csv'],
            key="csv_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Чтение файла
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Файл загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
                
                # Показать данные
                if st.checkbox("Показать первые 5 строк"):
                    st.dataframe(df.head())
                
                # Кнопка предсказания
                if st.button("🎯 Сделать предсказания", key="predict_csv"):
                    with st.spinner("Обрабатываю данные..."):
                        try:
                            # Преобразуем данные через препроцессор
                            X_processed = preprocessor.transform(df)
                            
                            # Если результат не DataFrame, преобразуем его
                            if not isinstance(X_processed, pd.DataFrame):
                                # Пробуем получить имена признаков
                                if hasattr(preprocessor, 'get_feature_names_out'):
                                    feature_names = preprocessor.get_feature_names_out()
                                    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
                                else:
                                    X_processed_df = pd.DataFrame(X_processed)
                            else:
                                X_processed_df = X_processed
                            
                            # Делаем предсказания
                            predictions = model.predict(X_processed_df)
                            
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
                            
                            # Показываем результаты
                            st.success("✅ Предсказания готовы!")
                            
                            # Статистика
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Средняя цена", f"${predictions.mean():,.0f}")
                            col2.metric("Минимальная цена", f"${predictions.min():,.0f}")
                            col3.metric("Максимальная цена", f"${predictions.max():,.0f}")
                            
                            # Таблица с результатами
                            st.subheader("Результаты предсказания")
                            st.dataframe(results.head(20))
                            
                            # Скачивание
                            csv_data = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать результаты (CSV)",
                                data=csv_data,
                                file_name="house_price_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при обработке данных: {str(e)[:200]}")
                            
                            # Отладочная информация
                            with st.expander("🔍 Детали ошибки"):
                                st.write(f"Тип X_processed: {type(X_processed)}")
                                if hasattr(X_processed, 'shape'):
                                    st.write(f"Форма X_processed: {X_processed.shape}")
                            
            except Exception as e:
                st.error(f"❌ Ошибка при чтении файла: {e}")
    
    with tab2:
        st.header("Ручной ввод параметров")
        
        # Простая форма для основных параметров
        with st.form("manual_input_form"):
            st.subheader("Основные параметры дома")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overall_qual = st.slider("Общее качество (1-10)", 1, 10, 7)
                gr_liv_area = st.number_input("Жилая площадь (кв.фут)", 500, 5000, 1500)
                total_bsmt_sf = st.number_input("Площадь подвала (кв.фут)", 0, 3000, 1000)
                year_built = st.number_input("Год постройки", 1900, 2024, 2000)
                
            with col2:
                lot_area = st.number_input("Площадь участка (кв.фут)", 1000, 50000, 10000)
                bedroom_abv_gr = st.slider("Количество спален", 0, 8, 3)
                full_bath = st.slider("Полных ванных", 0, 4, 2)
                fireplaces = st.slider("Камины", 0, 4, 1)
            
            # Категориальные признаки
            neighborhood = st.selectbox("Район", 
                ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'BrkSide'])
            
            kitchen_qual = st.selectbox("Качество кухни",
                ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            
            # Отправка формы
            submitted = st.form_submit_button("💰 Предсказать цену", use_container_width=True)
            
            if submitted:
                with st.spinner("Рассчитываю стоимость..."):
                    try:
                        # Создаем тестовые данные с основными параметрами
                        test_data = {
                            'MSSubClass': 60,
                            'MSZoning': 'RL',
                            'LotFrontage': 70.0,
                            'LotArea': lot_area,
                            'Street': 'Pave',
                            'Alley': 'NA',
                            'LotShape': 'Reg',
                            'LandContour': 'Lvl',
                            'Utilities': 'AllPub',
                            'LotConfig': 'Inside',
                            'LandSlope': 'Gtl',
                            'Neighborhood': neighborhood,
                            'Condition1': 'Norm',
                            'Condition2': 'Norm',
                            'BldgType': '1Fam',
                            'HouseStyle': '1Story',
                            'OverallQual': overall_qual,
                            'OverallCond': 5,
                            'YearBuilt': year_built,
                            'YearRemodAdd': year_built,
                            'RoofStyle': 'Gable',
                            'RoofMatl': 'CompShg',
                            'Exterior1st': 'VinylSd',
                            'Exterior2nd': 'VinylSd',
                            'MasVnrType': 'None',
                            'MasVnrArea': 0.0,
                            'ExterQual': 'TA',
                            'ExterCond': 'TA',
                            'Foundation': 'PConc',
                            'BsmtQual': 'TA',
                            'BsmtCond': 'TA',
                            'BsmtExposure': 'No',
                            'BsmtFinType1': 'Unf',
                            'BsmtFinSF1': 500.0,
                            'BsmtFinType2': 'Unf',
                            'BsmtFinSF2': 0.0,
                            'BsmtUnfSF': 500.0,
                            'TotalBsmtSF': total_bsmt_sf,
                            'Heating': 'GasA',
                            'HeatingQC': 'TA',
                            'CentralAir': 'Y',
                            'Electrical': 'SBrkr',
                            '1stFlrSF': 1200,
                            '2ndFlrSF': 0,
                            'LowQualFinSF': 0,
                            'GrLivArea': gr_liv_area,
                            'BsmtFullBath': 0,
                            'BsmtHalfBath': 0,
                            'FullBath': full_bath,
                            'HalfBath': 1,
                            'BedroomAbvGr': bedroom_abv_gr,
                            'KitchenAbvGr': 1,
                            'KitchenQual': kitchen_qual,
                            'TotRmsAbvGrd': 6,
                            'Functional': 'Typ',
                            'Fireplaces': fireplaces,
                            'FireplaceQu': 'NA',
                            'GarageType': 'Attchd',
                            'GarageYrBlt': year_built,
                            'GarageFinish': 'Unf',
                            'GarageCars': 2,
                            'GarageArea': 500,
                            'GarageQual': 'TA',
                            'GarageCond': 'TA',
                            'PavedDrive': 'Y',
                            'WoodDeckSF': 0,
                            'OpenPorchSF': 50,
                            'EnclosedPorch': 0,
                            '3SsnPorch': 0,
                            'ScreenPorch': 0,
                            'PoolArea': 0,
                            'PoolQC': 'NA',
                            'Fence': 'NA',
                            'MiscFeature': 'NA',
                            'MiscVal': 0,
                            'MoSold': 6,
                            'YrSold': 2024,
                            'SaleType': 'WD',
                            'SaleCondition': 'Normal'
                        }
                        
                        # Создаем DataFrame
                        df_test = pd.DataFrame([test_data])
                        
                        # Применяем препроцессор
                        X_processed = preprocessor.transform(df_test)
                        
                        # Если результат не DataFrame, преобразуем его
                        if not isinstance(X_processed, pd.DataFrame):
                            if hasattr(preprocessor, 'get_feature_names_out'):
                                feature_names = preprocessor.get_feature_names_out()
                                X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
                            else:
                                X_processed_df = pd.DataFrame(X_processed)
                        else:
                            X_processed_df = X_processed
                        
                        # Делаем предсказание
                        prediction = model.predict(X_processed_df)[0]
                        
                        # Показываем результат
                        st.success(f"## 🏡 Предсказанная цена: **${prediction:,.0f}**")
                        
                        # Информация о введенных параметрах
                        with st.expander("📊 Детали расчета"):
                            st.write(f"**Введенные параметры:**")
                            st.write(f"- Общее качество: {overall_qual}/10")
                            st.write(f"- Жилая площадь: {gr_liv_area} кв.футов")
                            st.write(f"- Площадь участка: {lot_area} кв.футов")
                            st.write(f"- Год постройки: {year_built}")
                            st.write(f"- Количество спален: {bedroom_abv_gr}")
                            st.write(f"- Район: {neighborhood}")
                            st.write(f"- Качество кухни: {kitchen_qual}")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)[:200]}")
                        
                        # Отладочная информация
                        with st.expander("🔍 Детали ошибки"):
                            if 'X_processed' in locals():
                                st.write(f"Тип X_processed: {type(X_processed)}")
                                if hasattr(X_processed, 'shape'):
                                    st.write(f"Форма X_processed: {X_processed.shape}")

else:
    st.warning("⚠️ Проверьте наличие файлов GB_model.pkl и preprocessor.pkl в папке")

# Футер
st.markdown("---")
st.markdown("""
### 📋 Требования к CSV файлу:
- Должен содержать все 79 признаков из оригинального датасета
- Категориальные признаки должны быть в строковом формате
- Числовые признаки должны быть в числовом формате

### 🔧 Техническая информация:
- Модель: GradientBoostingRegressor
- Препроцессор включает: CatBoostEncoder, StandardScaler
- Удалены 28 признаков из оригинального набора
""")