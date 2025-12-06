import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Настройка страницы
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Заголовок
st.title("🏠 House Price Predictor")
st.markdown("### Предсказание цен на дома с использованием Gradient Boosting")

# Функция для преобразования данных (такая же как в train_model)
def transform_new_data(X_new, transformers):
    """Преобразование новых данных"""
    numerical_cols = transformers['numerical_cols']
    categorical_cols = transformers['categorical_cols']
    numeric_imputer = transformers['numeric_imputer']
    scaler = transformers['scaler']
    cat_imputer = transformers['cat_imputer']
    label_encoders = transformers['label_encoders']
    
    # Числовые признаки
    X_num = numeric_imputer.transform(X_new[numerical_cols])
    X_num = scaler.transform(X_num)
    
    # Категориальные признаки
    X_cat = cat_imputer.transform(X_new[categorical_cols])
    
    # Кодирование
    for i, col in enumerate(categorical_cols):
        le = label_encoders[col]
        # Преобразуем с обработкой новых значений
        X_cat_col = X_cat[:, i]
        # Заменяем неизвестные значения на -1
        mask = np.isin(X_cat_col, le.classes_)
        X_cat_col[~mask] = -1
        # Для известных значений применяем transform
        known_values = X_cat_col[mask]
        if len(known_values) > 0:
            X_cat_col[mask] = le.transform(known_values)
        X_cat[:, i] = X_cat_col.astype(float)
    
    # Объединение
    return np.hstack([X_num, X_cat])

# Метрики
def calculate_metrics(y_true, y_pred):
    """Вычисление всех метрик"""
    metrics = {}
    
    # RMSE
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    
    # RMSLE
    try:
        metrics['RMSLE'] = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    except:
        metrics['RMSLE'] = np.nan
    
    # R²
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    if mask.any():
        metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['MAPE'] = np.nan
    
    return metrics

# Загрузка моделей
@st.cache_resource
def load_models():
    """Загрузка модели и transformers"""
    try:
        model = joblib.load('GB_model.pkl')
        transformers = joblib.load('transformers.pkl')
        feature_info = joblib.load('feature_info.pkl')
        return model, transformers, feature_info
    except Exception as e:
        st.error(f"❌ Ошибка загрузки моделей: {e}")
        st.info("Сначала обучите модель: python train_model_simple_fixed.py")
        return None, None, None

# Загружаем
model, transformers, feature_info = load_models()

if model and transformers and feature_info:
    st.success("✅ Модель и transformers успешно загружены!")
    
    # Сайдбар
    with st.sidebar:
        st.header("ℹ️ Информация о модели")
        st.write(f"**Тип:** GradientBoostingRegressor")
        st.write(f"**Количество деревьев:** {model.n_estimators}")
        st.write(f"**Глубина:** {model.max_depth}")
        st.write(f"**Скорость обучения:** {model.learning_rate:.3f}")
        
        st.header("📊 Признаки")
        st.write(f"Числовые: {len(feature_info['numerical_features'])}")
        st.write(f"Категориальные: {len(feature_info['categorical_features'])}")
        st.write(f"Всего: {len(feature_info['feature_names'])}")
    
    # Основной интерфейс
    tab1, tab2 = st.tabs(["📤 Загрузка CSV", "📝 Ручной ввод"])
    
    with tab1:
        st.header("Загрузите CSV файл для предсказания")
        
        uploaded_file = st.file_uploader(
            "Выберите CSV файл", 
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Файл загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
                
                # Проверка признаков
                missing_numeric = [col for col in feature_info['numerical_features'] 
                                  if col not in df.columns]
                missing_categorical = [col for col in feature_info['categorical_features'] 
                                      if col not in df.columns]
                
                if missing_numeric or missing_categorical:
                    st.warning("⚠️ Отсутствуют некоторые признаки")
                    if st.checkbox("Показать отсутствующие признаки"):
                        if missing_numeric:
                            st.write("**Числовые:**", missing_numeric)
                        if missing_categorical:
                            st.write("**Категориальные:**", missing_categorical)
                
                # Показать данные
                if st.checkbox("Показать первые 5 строк"):
                    st.dataframe(df.head())
                
                if st.button("🎯 Сделать предсказания", type="primary"):
                    with st.spinner("Обрабатываю данные..."):
                        try:
                            # Применяем преобразование
                            X_processed = transform_new_data(df, transformers)
                            
                            # Предсказания
                            predictions = model.predict(X_processed)
                            
                            # Результаты
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
                            
                            # Если есть реальные цены
                            if 'SalePrice' in df.columns:
                                y_true = df['SalePrice']
                                metrics = calculate_metrics(y_true, predictions)
                                
                                st.subheader("📈 Метрики качества")
                                cols = st.columns(5)
                                metric_data = [
                                    ("RMSE", f"${metrics['RMSE']:,.0f}"),
                                    ("MAE", f"${metrics['MAE']:,.0f}"),
                                    ("R²", f"{metrics['R2']:.4f}"),
                                    ("MAPE", f"{metrics['MAPE']:.1f}%" if not np.isnan(metrics['MAPE']) else "N/A"),
                                    ("RMSLE", f"{metrics['RMSLE']:.4f}" if not np.isnan(metrics['RMSLE']) else "N/A")
                                ]
                                
                                for i, (name, value) in enumerate(metric_data):
                                    with cols[i]:
                                        st.metric(name, value)
                            
                            # Статистика
                            st.subheader("📊 Статистика предсказаний")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Средняя", f"${predictions.mean():,.0f}")
                            with col2:
                                st.metric("Медиана", f"${np.median(predictions):,.0f}")
                            with col3:
                                st.metric("Минимум", f"${predictions.min():,.0f}")
                            with col4:
                                st.metric("Максимум", f"${predictions.max():,.0f}")
                            
                            # Таблица
                            st.subheader("📋 Результаты (первые 20)")
                            st.dataframe(results.head(20))
                            
                            # Скачивание
                            csv_data = results.to_csv(index=False)
                            st.download_button(
                                "📥 Скачать все результаты",
                                csv_data,
                                "predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка обработки: {str(e)[:200]}")
                            
            except Exception as e:
                st.error(f"❌ Ошибка чтения файла: {e}")
    
    with tab2:
        st.header("Ручной ввод параметров")
        
        with st.form("house_form"):
            st.subheader("Основные параметры")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overall_qual = st.slider("Общее качество (1-10)", 1, 10, 7)
                gr_liv_area = st.number_input("Жилая площадь", 500, 5000, 1500)
                total_bsmt_sf = st.number_input("Площадь подвала", 0, 3000, 1000)
                year_built = st.number_input("Год постройки", 1900, 2024, 2000)
                
            with col2:
                lot_area = st.number_input("Площадь участка", 1000, 50000, 10000)
                garage_cars = st.slider("Машиномест в гараже", 0, 4, 2)
                full_bath = st.slider("Полных ванных", 0, 4, 2)
                fireplaces = st.slider("Камины", 0, 4, 1)
            
            # Категориальные признаки
            with st.expander("📋 Дополнительные параметры"):
                mszoning = st.selectbox("Зонирование", 
                    ['RL', 'RM', 'C (all)', 'FV', 'RH'])
                neighborhood = st.selectbox("Район", 
                    ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 
                     'Gilbert', 'NridgHt', 'BrkSide'])
                kitchen_qual = st.selectbox("Качество кухни",
                    ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            
            submitted = st.form_submit_button("💰 Предсказать цену")
            
            if submitted:
                with st.spinner("Рассчитываю..."):
                    try:
                        # Создаем словарь
                        house_data = {}
                        
                        # Заполняем все числовые признаки
                        for col in feature_info['numerical_features']:
                            if col == 'OverallQual':
                                house_data[col] = overall_qual
                            elif col == 'GrLivArea':
                                house_data[col] = gr_liv_area
                            elif col == 'TotalBsmtSF':
                                house_data[col] = total_bsmt_sf
                            elif col == 'YearBuilt':
                                house_data[col] = year_built
                            elif col == 'LotArea':
                                house_data[col] = lot_area
                            elif col == 'GarageCars':
                                house_data[col] = garage_cars
                            elif col == 'FullBath':
                                house_data[col] = full_bath
                            elif col == 'Fireplaces':
                                house_data[col] = fireplaces
                            else:
                                # Значения по умолчанию для остальных числовых признаков
                                if col == 'OverallCond':
                                    house_data[col] = 5
                                elif col == 'YearRemodAdd':
                                    house_data[col] = year_built
                                elif 'Area' in col or 'SF' in col:
                                    house_data[col] = 0
                                elif 'Bath' in col:
                                    house_data[col] = 0
                                else:
                                    house_data[col] = 0
                        
                        # Заполняем категориальные признаки
                        for col in feature_info['categorical_features']:
                            if col == 'MSZoning':
                                house_data[col] = mszoning
                            elif col == 'Neighborhood':
                                house_data[col] = neighborhood
                            elif col == 'KitchenQual':
                                house_data[col] = kitchen_qual
                            else:
                                # Значения по умолчанию
                                if col == 'CentralAir':
                                    house_data[col] = 'Y'
                                elif col == 'PavedDrive':
                                    house_data[col] = 'Y'
                                elif col == 'SaleCondition':
                                    house_data[col] = 'Normal'
                                elif col == 'SaleType':
                                    house_data[col] = 'WD'
                                elif col == 'BsmtQual':
                                    house_data[col] = 'TA'
                                elif col == 'GarageType':
                                    house_data[col] = 'Attchd'
                                else:
                                    house_data[col] = 'NA'
                        
                        # DataFrame
                        df_input = pd.DataFrame([house_data])
                        
                        # Преобразование и предсказание
                        X_processed = transform_new_data(df_input, transformers)
                        prediction = model.predict(X_processed)[0]
                        
                        # Результат
                        st.success(f"## 🏡 Предсказанная цена: **${prediction:,.0f}**")
                        
                        # Детали
                        with st.expander("📊 Использованные параметры"):
                            st.write("**Основные:**")
                            cols = st.columns(2)
                            with cols[0]:
                                st.write(f"- Общее качество: {overall_qual}/10")
                                st.write(f"- Жилая площадь: {gr_liv_area} кв.футов")
                                st.write(f"- Площадь подвала: {total_bsmt_sf} кв.футов")
                                st.write(f"- Год постройки: {year_built}")
                            with cols[1]:
                                st.write(f"- Площадь участка: {lot_area} кв.футов")
                                st.write(f"- Машиномест в гараже: {garage_cars}")
                                st.write(f"- Полных ванных: {full_bath}")
                                st.write(f"- Камины: {fireplaces}")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)[:200]}")

else:
    st.warning("⚠️ Сначала обучите модель, запустив train_model_simple_fixed.py")

# Инструкция
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 Как использовать:

1. **Обучите модель:**
```bash
python train_model_simple_fixed.py""")