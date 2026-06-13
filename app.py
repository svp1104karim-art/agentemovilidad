import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="TránsitoLegal AI - Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz premium
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradiente de fondo y tarjetas */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(90, 92, 106, 0.05) 0%, rgba(32, 35, 47, 0.05) 90%);
    }
    
    /* Tarjetas personalizadas */
    .metric-card {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #4F46E5;
        transition: transform 0.3s ease;
        margin-bottom: 20px;
        color: #1F2937;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 14px;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 28px;
        color: #1F2937;
        font-weight: 700;
        margin-top: 8px;
    }
    
    /* Estilos del Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }
    
    /* Títulos e Headers */
    .main-title {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 40px;
        margin-bottom: 10px;
    }
    
    /* Badges de Diagnóstico */
    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success { background-color: #D1FAE5; color: #065F46; }
    .badge-error { background-color: #FEE2E2; color: #991B1B; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
    .badge-info { background-color: #DBEAFE; color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

# --- CARGAR DATOS Y MODELO ---
@st.cache_data
def cargar_datos_y_modelo():
    try:
        # Carga el dataset con UTF-8
        df = pd.read_csv('dataset.csv', encoding='utf-8')
    except FileNotFoundError:
        st.error("🚨 El archivo 'dataset.csv' no se encuentra en la raíz del proyecto.")
        st.stop()

    df['FECHA_INFRACCION'] = pd.to_datetime(df['FECHA_INFRACCION'], dayfirst=True)
    fecha_referencia = pd.to_datetime('01/01/2018')
    df['DIAS_DESDE_REFERENCIA'] = (df['FECHA_INFRACCION'] - fecha_referencia).dt.days

    # Preparar entrenamiento
    features = ['DIAS_DESDE_REFERENCIA', 'CLASE_VEHI', 'COD_INFRACCION']
    X = df[features].copy()
    y = df['INCONSISTENCIA_JUSTIFICACION']

    le_clase = LabelEncoder()
    le_codigo = LabelEncoder()

    X['CLASE_VEHI_NUM'] = le_clase.fit_transform(X['CLASE_VEHI'])
    X['COD_INFRACCION_NUM'] = le_codigo.fit_transform(X['COD_INFRACCION'])
    X = X.drop(['CLASE_VEHI', 'COD_INFRACCION'], axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return df, model, le_clase, le_codigo

df, model, le_clase, le_codigo = cargar_datos_y_modelo()

# --- SIDEBAR NAVEGACIÓN ---
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-lineal-color-flatart-icons/128/external-justice-law-and-justice-flatart-icons-lineal-color-flatart-icons.png", width=80)
    st.markdown("## TránsitoLegal AI")
    st.markdown("Sistema inteligente de defensa y diagnóstico legal de comparendos de tránsito.")
    st.write("---")
    
    opcion = st.radio(
        "Navegación",
        ["📊 Dashboard de Control", "🔍 Diagnóstico de Comparendos", "⚖️ Guía Legal y Plantillas"]
    )
    
    st.write("---")
    st.info("💡 **Versión:** 1.2.0 (Prototipo Web)\n\nImpulsado por Machine Learning para detectar inconsistencias en comparendos colombianos.")

# --- PÁGINA 1: DASHBOARD ---
if opcion == "📊 Dashboard de Control":
    st.markdown("<h1 class='main-title'>📊 Dashboard de Control e Inconsistencias</h1>", unsafe_allow_html=True)
    st.write("Análisis general del dataset cargado para detectar irregularidades en la imposición de multas.")
    
    # Cálculos estadísticos
    total_comparendos = len(df)
    inconsistencias = df[df['INCONSISTENCIA_JUSTIFICACION'] != 'SIN INCONSISTENCIAS']
    num_inconsistencias = len(inconsistencias)
    tasa_irregularidad = (num_inconsistencias / total_comparendos) * 100
    
    # Total en multas e inconsistentes (Dinero)
    total_pesos = df['VALOR_SANCION'].sum()
    total_ahorro_potencial = inconsistencias['VALOR_SANCION'].sum()

    # Layout de Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #6366F1;">
            <div class="metric-title">Comparendos Totales</div>
            <div class="metric-value">{total_comparendos}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #EF4444;">
            <div class="metric-title">Con Inconsistencias</div>
            <div class="metric-value">{num_inconsistencias} <span style="font-size:16px; color:#EF4444;">({tasa_irregularidad:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10B981;">
            <div class="metric-title">Recaudo Total Registrado</div>
            <div class="metric-value">${total_pesos:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #F59E0B;">
            <div class="metric-title">Potencial de Ahorro / Impugnación</div>
            <div class="metric-value">${total_ahorro_potencial:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Gráficos y Tablas
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("⚠️ Tipo de Inconsistencias Detectadas")
        inconsistencia_counts = df['INCONSISTENCIA_JUSTIFICACION'].value_counts()
        # Excluir 'SIN INCONSISTENCIAS' para visualizar mejor los errores
        inc_only = inconsistencia_counts[inconsistencia_counts.index != 'SIN INCONSISTENCIAS']
        if not inc_only.empty:
            st.bar_chart(inc_only)
        else:
            st.success("No se han detectado inconsistencias en el dataset.")
            
    with c2:
        st.subheader("🚗 Distribución por Clase de Vehículo")
        vehiculos_counts = df['CLASE_VEHI'].value_counts()
        st.bar_chart(vehiculos_counts)

    st.subheader("📋 Vista Rápida del Registro de Comparendos")
    # Mostrar tabla estilizada con columnas específicas
    st.dataframe(
        df[['ID_COMPARENDO', 'FECHA_INFRACCION', 'PLACA', 'COD_INFRACCION', 'CLASE_VEHI', 'VALOR_SANCION', 'ESTADO', 'INCONSISTENCIA_JUSTIFICACION']],
        use_container_width=True
    )

# --- PÁGINA 2: DIAGNÓSTICO ---
elif opcion == "🔍 Diagnóstico de Comparendos":
    st.markdown("<h1 class='main-title'>🔍 Analizador Inteligente de Comparendos</h1>", unsafe_allow_html=True)
    st.write("Selecciona un comparendo existente o introduce datos de prueba para que el modelo de Inteligencia Artificial evalúe la viabilidad de una impugnación.")
    
    tabs = st.tabs(["🗂️ Seleccionar del Dataset", "✍️ Introducir Manualmente"])
    
    with tabs[0]:
        st.subheader("Analizar Comparendo Registrado")
        comparendo_seleccionado = st.selectbox(
            "Selecciona el ID del Comparendo del dataset:",
            df['ID_COMPARENDO'].tolist()
        )
        
        if st.button("Ejecutar Análisis de IA", key="btn_dataset"):
            caso = df[df['ID_COMPARENDO'] == comparendo_seleccionado].iloc[0]
            
            # Preparar predicción
            fecha = caso['FECHA_INFRACCION']
            fecha_referencia = pd.to_datetime('01/01/2018')
            dias = (fecha - fecha_referencia).days
            
            try:
                clase = le_clase.transform([caso['CLASE_VEHI']])[0]
                codigo = le_codigo.transform([caso['COD_INFRACCION']])[0]
                datos_prediccion = [[dias, clase, codigo]]
                
                prediccion = model.predict(datos_prediccion)[0]
                probabilidades = model.predict_proba(datos_prediccion)[0]
                confianza = max(probabilidades)
            except Exception as e:
                prediccion = caso['INCONSISTENCIA_JUSTIFICACION']
                confianza = 1.0
                
            # Renderizar interfaz de resultados
            st.divider()
            st.subheader("📋 Ficha de Comparendo Analizada")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**ID:** {caso['ID_COMPARENDO']}")
                st.write(f"📅 **Fecha de Infracción:** {caso['FECHA_INFRACCION'].strftime('%d/%m/%Y')}")
                st.write(f"🚗 **Vehículo:** {caso['CLASE_VEHI']} ({caso['PLACA']})")
                st.write(f"📍 **Lugar:** {caso['LUGAR_INFRACCION']}")
            with col2:
                st.write(f"🛑 **Código Infracción:** {caso['COD_INFRACCION']}")
                st.write(f"💰 **Valor Sanción:** ${caso['VALOR_SANCION']:,.0f}")
                st.write(f"📌 **Estado Actual:** {caso['ESTADO']}")
            
            st.divider()
            st.subheader("🤖 Diagnóstico y Viabilidad Legal")
            
            # Tarjeta de diagnóstico basada en predicción
            if prediccion == 'SIN INCONSISTENCIAS':
                st.success(f"✅ **Diagnóstico de la IA:** {prediccion}")
                st.metric("Confianza de la IA", f"{confianza * 100:.1f}%")
                st.markdown("""
                ### 💡 Recomendación de TránsitoLegal:
                El comparendo cumple con los requisitos iniciales. **Recomendamos realizar el pago dentro de los términos legales** para acceder al 50% de descuento por pronto pago e inscribirse al curso pedagógico.
                """)
            elif 'PRESCRITO' in prediccion:
                st.error(f"⚖️ **Diagnóstico de la IA:** {prediccion}")
                st.metric("Confianza de la IA", f"{confianza * 100:.1f}%")
                st.markdown("""
                ### 💡 Recomendación de TránsitoLegal:
                **¡Oportunidad de Anulación!** De acuerdo con el Art. 159 del Código Nacional de Tránsito, la sanción ha superado el tiempo legal máximo de cobro (3 años).
                * **Acción sugerida:** Radicar inmediatamente un **Derecho de Petición por Prescripción** ante la Secretaría de Movilidad correspondiente. Puedes descargar la plantilla en la pestaña **Guía Legal y Plantillas**.
                """)
            elif 'ERROR' in prediccion:
                st.warning(f"⚠️ **Diagnóstico de la IA:** {prediccion}")
                st.metric("Confianza de la IA", f"{confianza * 100:.1f}%")
                st.markdown(f"""
                ### 💡 Recomendación de TránsitoLegal:
                **¡Inconsistencia Detectada!** El modelo detectó un error técnico en el procedimiento: *{prediccion}*.
                * **Acción sugerida:** Presentar una **Impugnación de Comparendo** en audiencia pública argumentando falla formal de la autoridad (violación al debido proceso legal, Ley 1437 de 2011).
                """)
            else:
                st.info(f"📋 **Diagnóstico de la IA:** {prediccion}")
                st.write("Por favor, consulta los detalles con un asesor legal.")

    with tabs[1]:
        st.subheader("Simular un Comparendo Nuevo")
        c1, c2, c3 = st.columns(3)
        with c1:
            fecha_input = st.date_input("Fecha de la Infracción", datetime.date.today())
            valor_input = st.number_input("Valor de la Multa ($)", min_value=0, value=458000, step=50000)
        with c2:
            clase_input = st.selectbox("Clase de Vehículo", df['CLASE_VEHI'].unique())
            placa_input = st.text_input("Placa del Vehículo", "XYZ987")
        with c3:
            codigo_input = st.selectbox("Código de Infracción", df['COD_INFRACCION'].unique())
            estado_input = st.selectbox("Estado", ["PENDIENTE", "EN PROCESO", "PAGADO"])
            
        if st.button("Predecir Viabilidad de Impugnación", key="btn_manual"):
            fecha_dt = pd.to_datetime(fecha_input)
            fecha_referencia = pd.to_datetime('01/01/2018')
            dias_input = (fecha_dt - fecha_referencia).days
            
            try:
                # Transformar entradas manuales
                clase_num = le_clase.transform([clase_input])[0]
                codigo_num = le_codigo.transform([codigo_input])[0]
                datos_input = [[dias_input, clase_num, codigo_num]]
                
                pred_input = model.predict(datos_input)[0]
                prob_input = model.predict_proba(datos_input)[0]
                confianza_input = max(prob_input)
            except Exception:
                pred_input = "SIN INCONSISTENCIAS (El modelo requiere datos más precisos)"
                confianza_input = 0.50
                
            st.divider()
            st.subheader("🤖 Diagnóstico de IA Simulado")
            st.metric("Nivel de Confianza", f"{confianza_input * 100:.1f}%")
            
            if "SIN INCONSISTENCIAS" in pred_input:
                st.success(f"✅ **Resultado:** {pred_input}")
                st.write("El modelo simula que el comparendo cumple los requisitos formales de la ley colombiana.")
            else:
                st.warning(f"⚠️ **Resultado:** {pred_input}")
                st.write("El modelo simula una alta probabilidad de inconsistencias procedimentales. Es viable una reclamación.")

# --- PÁGINA 3: GUÍA LEGAL ---
elif opcion == "⚖️ Guía Legal y Plantillas":
    st.markdown("<h1 class='main-title'>⚖️ Biblioteca Legal y Plantillas de Reclamación</h1>", unsafe_allow_html=True)
    st.write("Recursos prácticos basados en la legislación colombiana (Ley 769 de 2002) para impugnar tus multas de tránsito.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📄 Plantilla: Derecho de Petición por Prescripción")
        st.info("Utiliza esta plantilla si tu comparendo tiene más de 3 años de antigüedad y el organismo de tránsito no ha emitido mandamiento de pago.")
        
        peticion_template = """CIUDAD Y FECHA: [Inserta Ciudad y Fecha]

Señores:
ORGANISMO DE TRÁNSITO Y MOVILIDAD DE [Inserta Ciudad]
E. S. D.

Asunto: DERECHO DE PETICIÓN EN INTERÉS PARTICULAR - SOLICITUD DE DECLARATORIA DE PRESCRIPCIÓN DE COMPARENDO E INFRACCIÓN DE TRÁNSITO

Yo, [INSERTA TU NOMBRE], identificado(a) con cédula de ciudadanía No. [INSERTA TU CÉDULA], en ejercicio del derecho constitucional fundamental consagrado en el artículo 23 de la Constitución Política de Colombia y regulado por la Ley 1755 de 2015, comparezco ante ustedes con el fin de formular la siguiente petición respetuosa:

1. PETICIONES
1.1. Solicito se declare la PRESCRIPCIÓN de la acción de cobro y de la sanción derivada del comparendo No. [INSERTA NÚMERO DE COMPARENDO], impuesto el día [INSERTA FECHA], sobre el vehículo de placa [INSERTA PLACA], de conformidad con lo establecido en el Artículo 159 de la Ley 769 de 2002 (Código Nacional de Tránsito).
1.2. Como consecuencia de la anterior declaración, se ordene la eliminación y retiro definitivo del citado comparendo del sistema oficial de información (SIMIT / RUNT).

2. FUNDAMENTOS DE DERECHO
- Artículo 159 de la Ley 769 de 2002: Establece que las sanciones impuestas por infracciones a las normas de tránsito prescriben en tres (3) años contados a partir de la ocurrencia del hecho.
- Artículo 818 del Estatuto Tributario y jurisprudencia constitucional sobre el debido proceso administrativo.

Quedo atento a su respuesta dentro de los términos de ley.

Atentamente,
_____________________________
FIRMA:
C.C.:
Correo Electrónico:
Teléfono:"""
        
        st.text_area("Copiar plantilla", peticion_template, height=300)
        
    with c2:
        st.subheader("📚 Marco Jurídico Clave")
        st.markdown("""
        **1. Prescripción (Art. 159 Ley 769/02)**
        El tiempo límite del cobro del comparendo es de **3 años** desde la infracción. Si el organismo expide resolución sancionatoria, se interrumpe y cuenta otros 3 años.
        
        **2. Notificación en Fotomultas (Sentencia STC-9884/2021)**
        Las fotomultas deben notificarse dentro de los **13 días hábiles** siguientes a su imposición. De lo contrario, se viola el debido proceso y la multa es nula.
        
        **3. Caducidad de la Acción (Art. 161)**
        La audiencia pública debe citarse en un plazo no mayor a **1 año** a partir de la infracción, de lo contrario la administración pierde facultad para sancionar.
        """)
        st.success("👨‍⚖️ **Nota:** Las plantillas son guías de referencia. Recomendamos consultar a un abogado para casos complejos.")
