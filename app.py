import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación", layout="wide")

st.title("Predicción de Aprobación de Curso")
st.markdown("### Desarrollado por: Brandon Steven Calzada Urrea")

st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado. Puedes ingresar los datos manualmente o subir un archivo Excel.")

# 1. Cargar artefactos (con caché para mejor rendimiento)
@st.cache_resource
def load_artifacts():
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
    return one_hot_transformer, scaler, model

try:
    one_hot_transformer, scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}. Asegúrate de que los archivos .joblib existan.")
    st.stop()

# Función centralizada para procesar datos y predecir
def procesar_y_predecir(df_input):
    df_procesado = df_input.copy()

    # Convertir a One-Hot
    if isinstance(one_hot_transformer, list):
        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
        for col_name in si_columnas_one_hot:
            valor_esperado = col_name.replace('Felder_', '')
            if 'Felder' in df_procesado.columns:
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
            else:
                df_procesado[col_name] = 0
    else:
        df_encoded = pd.get_dummies(df_procesado[['Felder']])
        df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
        si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

    # Eliminar variable original
    df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

    # Asegurar que existan las columnas One-Hot
    if isinstance(one_hot_transformer, list):
        for col in si_columnas_one_hot:
            if col not in df_procesado.columns:
                df_procesado[col] = 0

    # Escalar Examen_admisión
    if 'Examen_admisión' in df_procesado.columns:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
    else:
        st.error("Falta la columna 'Examen_admisión' en los datos.")
        st.stop()

    # Reordenar y filtrar columnas según el modelo
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    df_final = df_procesado[columnas_ordenadas]

    # Predicción
    predicciones = model.predict(df_final)
    return df_final, predicciones

# Crear pestañas para las dos opciones
tab1, tab2 = st.tabs(["Ingreso Manual", "Subir Archivo Excel"])

with tab1:
    st.header("Datos de Entrada Manual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

    if st.button("Realizar Predicción Manual"):
        try:
            df_manual = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
            df_final, prediccion = procesar_y_predecir(df_manual)

            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_final)

            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

with tab2:
    st.header("Predicción por Lotes (Excel)")
    archivo_subido = st.file_uploader("Sube un archivo Excel que contenga las columnas 'Felder' y 'Examen_admisión'", type=["xlsx", "xls"])

    if archivo_subido is not None:
        try:
            df_excel = pd.read_excel(archivo_subido)
            st.write("**Vista previa de los datos cargados:**")
            st.dataframe(df_excel.head())

            # Validar que existan las columnas mínimas
            if 'Felder' not in df_excel.columns or 'Examen_admisión' not in df_excel.columns:
                st.error("El archivo Excel debe contener al menos las columnas 'Felder' y 'Examen_admisión'.")
            else:
                if st.button("Realizar Predicción desde Archivo"):
                    df_final, predicciones = procesar_y_predecir(df_excel)

                    # Crear DataFrame de resultados
                    df_resultados = df_excel.copy()
                    df_resultados['Nota_Final_Predicha'] = predicciones

                    st.success("¡Predicciones realizadas con éxito!")
                    st.write("**Resultados:**")
                    st.dataframe(df_resultados)

                    # Botón para descargar resultados
                    @st.cache_data
                    def convertir_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convertir_df(df_resultados)
                    st.download_button(
                        label="Descargar resultados como CSV",
                        data=csv,
                        file_name='predicciones_resultados.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
