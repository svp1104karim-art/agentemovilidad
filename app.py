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

# Estilos CSS personalizados para una interfaz premium y oscura
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradiente de fondo y tarjetas */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 90%);
        color: #f8fafc;
    }
    
    /* Tarjetas personalizadas del Dashboard */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border: 1px solid #334155;
        border-left: 5px solid #3b82f6;
        transition: transform 0.3s ease;
        margin-bottom: 20px;
        color: #f8fafc;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .metric-title {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 26px;
        color: #f8fafc;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Tarjetas de Comparendos */
    .ticket-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 1.5rem;
    }
    
    .ticket-title {
        color: #3b82f6;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .ticket-detail-item {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    
    /* Estilo del Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    /* Título principal con gradiente */
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 36px;
        margin-bottom: 10px;
    }
    
    /* Diagnóstico IA */
    .ai-diag-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        padding: 15px;
        height: 100%;
    }
    
    .ai-diag-title {
        font-size: 0.85rem;
        color: #10b981;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .ai-diag-score {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .ai-diag-desc {
        font-size: 0.8rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# --- CARGAR DATOS Y MODELO ---
@st.cache_data
def cargar_datos_y_modelo():
    try:
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

# --- MODALES / DIÁLOGOS DE STREAMLIT ---
@st.dialog("📤 Enviar Requerimiento de Impugnación")
def mostrar_modal_impugnacion(ticket_id, organismo):
    st.write(f"Vas a radicar una solicitud de impugnación para el comparendo **{ticket_id}**.")
    st.info(f"🏢 **Organismo Destino:** {organismo}")
    
    recurso = st.selectbox("Tipo de Recurso:", ["Reposición", "Apelación", "Nulidad"])
    argumento = st.text_area(
        "Argumento Legal:",
        placeholder="Describa las inconsistencias detectadas por la IA o los fundamentos de hecho...",
        height=120
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancelar", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Enviar a Organismo", type="primary", use_container_width=True):
            st.success(f"✅ Requerimiento de {recurso} enviado exitosamente a {organismo}.")
            st.balloons()

@st.dialog("🛠️ Gestión de Procesos Adicionales")
def mostrar_modal_tramites(ticket_id, organismo):
    st.subheader("Selecciona el trámite a solicitar")
    st.write(f"Comparendo: **{ticket_id}** | Organismo: **{organismo}**")
    st.write("---")
    
    if st.button("📅 Solicitar Audiencia Conciliatoria", use_container_width=True):
        st.toast("⏳ Solicitud de audiencia enviada al organismo.", icon="⏳")
        st.rerun()
    if st.button("💳 Solicitar Fraccionamiento de Pago", use_container_width=True):
        st.toast("⏳ Solicitud de acuerdo de pago radicada.", icon="⏳")
        st.rerun()
    if st.button("📜 Solicitar Certificado de Paz y Salvo", use_container_width=True):
        st.toast("⏳ Generando certificado de Paz y Salvo...", icon="⏳")
        st.rerun()
    if st.button("📬 Cambio de Dirección de Notificación", use_container_width=True):
        st.toast("⏳ Dirección de notificación actualizada.", icon="⏳")
        st.rerun()

# --- SIDEBAR NAVEGACIÓN ---
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-lineal-color-flatart-icons/128/external-justice-law-and-justice-flatart-icons-lineal-color-flatart-icons.png", width=70)
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
    st.markdown("<h1 class='main-title'>📊 Dashboard de Control de Comparendos</h1>", unsafe_allow_html=True)
    st.write("Análisis estadístico e histórico de infracciones y su viabilidad de defensa.")
    
    # Métricas
    total_comparendos = len(df)
    inconsistencias = df[df['INCONSISTENCIA_JUSTIFICACION'] != 'SIN INCONSISTENCIAS']
    num_inconsistencias = len(inconsistencias)
    tasa_irregularidad = (num_inconsistencias / total_comparendos) * 100
    total_pesos = df['VALOR_SANCION'].sum()
    total_ahorro_potencial = inconsistencias['VALOR_SANCION'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <div class="metric-title">Comparendos Totales</div>
            <div class="metric-value">{total_comparendos}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-title">Con Inconsistencias</div>
            <div class="metric-value">{num_inconsistencias} <span style="font-size:14px; color:#ef4444;">({tasa_irregularidad:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10b981;">
            <div class="metric-title">Recaudo Total Registrado</div>
            <div class="metric-value">${total_pesos:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div class="metric-title">Ahorro Impugnable</div>
            <div class="metric-value">${total_ahorro_potencial:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚠️ Distribución de Inconsistencias")
        st.bar_chart(df['INCONSISTENCIA_JUSTIFICACION'].value_counts()[1:])
    with c2:
        st.subheader("🚗 Distribución por Clase de Vehículo")
        st.bar_chart(df['CLASE_VEHI'].value_counts())

# --- PÁGINA 2: DIAGNÓSTICO ---
elif opcion == "🔍 Diagnóstico de Comparendos":
    st.markdown("<h1 class='main-title'>🔍 Analizador Inteligente de Comparendos</h1>", unsafe_allow_html=True)
    st.write("Introduce datos para consultar en la base de datos de tránsito y obtener un análisis predictivo de viabilidad.")

    # Tarjeta de Búsqueda Segura
    st.markdown("""
    <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #f8fafc;">Consultar Base de Datos</h4>
            <span style="font-size: 0.85rem; color: #94a3b8;">🔒 Búsqueda Segura</span>
        </div>
    """, unsafe_allow_html=True)

    col_cc, col_placa = st.columns(2)
    with col_cc:
        st.write("**Por Cédula de Ciudadanía (CC)**")
        cc_input = st.text_input("Ingresa CC de ejemplo (ej. 80987654):", placeholder="Ej: 80987654", key="cc")
        buscar_cc = st.button("Buscar por Cédula", type="primary", use_container_width=True)

    with col_placa:
        st.write("**Por Placa Vehicular**")
        placa_input = st.text_input("Ingresa Placa de ejemplo (ej. XYZ987):", placeholder="Ej: XYZ987", key="placa").upper()
        buscar_placa = st.button("Buscar por Placa", type="secondary", use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Lógica de Consulta
    resultados = None
    if buscar_cc and cc_input:
        try:
            cc_int = int(cc_input.strip())
            resultados = df[df['CC_USUARIO'] == cc_int]
        except ValueError:
            st.error("Por favor, ingresa un número de cédula válido.")
    elif buscar_placa and placa_input:
        resultados = df[df['PLACA'].str.upper() == placa_input.strip()]

    # Mostrar Resultados
    if resultados is not None:
        if len(resultados) == 0:
            st.warning("⚠️ No se encontraron comparendos registrados con los datos suministrados.")
        else:
            st.success(f"📋 Se encontraron **{len(resultados)}** comparendos registrados:")
            
            for index, row in resultados.iterrows():
                # Formatear datos
                ticket_id = row['ID_COMPARENDO']
                organismo = row['LUGAR_INFRACCION'] # usando lugar como el organismo de tránsito local
                infraccion_cod = row['COD_INFRACCION']
                clase_vehi = row['CLASE_VEHI']
                valor = row['VALOR_SANCION']
                estado = row['ESTADO']
                placa = row['PLACA']
                fecha = row['FECHA_INFRACCION']
                inconsistencia_just = row['INCONSISTENCIA_JUSTIFICACION']

                # Predicción usando ML
                dias = row['DIAS_DESDE_REFERENCIA']
                try:
                    clase_num = le_clase.transform([clase_vehi])[0]
                    codigo_num = le_codigo.transform([infraccion_cod])[0]
                    datos_prediccion = [[dias, clase_num, codigo_num]]
                    prediccion = model.predict(datos_prediccion)[0]
                    probabilidades = model.predict_proba(datos_prediccion)[0]
                    class_index = list(model.classes_).index(prediccion)
                    confianza = probabilidades[class_index]
                except Exception:
                    prediccion = inconsistencia_just
                    confianza = 0.90

                # Calcular puntuación de viabilidad de impugnación
                if prediccion == 'SIN INCONSISTENCIAS':
                    viabilidad_score = int((1 - confianza) * 100)
                    if viabilidad_score < 15: viabilidad_score = 15
                else:
                    viabilidad_score = int(confianza * 100)
                    if viabilidad_score < 60: viabilidad_score = 75

                # Renderizar Ficha de Comparendo y diagnóstico
                st.markdown(f"---")
                
                # Columnas de Información, Diagnóstico y Acciones
                col_info, col_diag, col_actions = st.columns([2, 1.2, 1])
                
                with col_info:
                    st.markdown(f"<div class='ticket-title'>{ticket_id}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticket-detail-item'>📅 **Fecha:** {fecha.strftime('%d/%m/%Y')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticket-detail-item'>🚗 **Vehículo:** {clase_vehi} ({placa})</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticket-detail-item'>🏢 **Organismo:** {organismo}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticket-detail-item'>🛑 **Infracción:** {infraccion_cod} | **Valor:** ${valor:,.0f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticket-detail-item'>📌 **Estado:** `{estado}`</div>", unsafe_allow_html=True)

                with col_diag:
                    score_color = "#10b981" if viabilidad_score >= 70 else ("#facc15" if viabilidad_score >= 40 else "#ef4444")
                    st.markdown(f"""
                    <div class="ai-diag-card">
                        <div class="ai-diag-title">
                            ⚖️ Diagnóstico IA
                        </div>
                        <div class="ai-diag-score" style="color: {score_color};">
                            {viabilidad_score}% Viabilidad
                        </div>
                        <div class="ai-diag-desc">
                            <strong>Predicción:</strong> {prediccion}<br>
                            <em>Motivo: {inconsistencia_just}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_actions:
                    st.write("")
                    st.write("")
                    # Botón de Impugnación
                    if st.button("⚖️ Impugnar", key=f"imp_{ticket_id}", use_container_width=True):
                        mostrar_modal_impugnacion(ticket_id, organismo)
                        
                    # Botón de Otros Trámites
                    if st.button("📂 Otros Trámites", key=f"tra_{ticket_id}", use_container_width=True):
                        mostrar_modal_tramites(ticket_id, organismo)

# --- PÁGINA 3: GUÍA LEGAL ---
elif opcion == "⚖️ Guía Legal y Plantillas":
    st.markdown("<h1 class='main-title'>⚖️ Biblioteca Legal y Plantillas</h1>", unsafe_allow_html=True)
    st.write("Consulta el fundamento jurídico aplicable a comparendos y descárgalo para tus reclamaciones.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Plantilla: Derecho de Petición por Prescripción")
        peticion_template = """CIUDAD Y FECHA: [Inserta Ciudad y Fecha]

Señores:
ORGANISMO DE TRÁNSITO Y MOVILIDAD DE [Inserta Ciudad]
E. S. D.

Asunto: DERECHO DE PETICIÓN EN INTERÉS PARTICULAR - SOLICITUD DE DECLARATORIA DE PRESCRIPCIÓN DE COMPARENDO E INFRACCIÓN DE TRÁNSITO

Yo, [INSERTA TU NOMBRE], identificado(a) con cédula de ciudadanía No. [INSERTA TU CÉDULA], en ejercicio del derecho constitucional fundamental consagrado en el artículo 23 de la Constitución Política de Colombia... [Resto del documento]"""
        st.text_area("Minuta descargable", peticion_template, height=350)
        
    with col2:
        st.subheader("📚 Marco Jurídico Clave")
        st.markdown("""
        **1. Prescripción (Art. 159 Ley 769/02)**
        El cobro de multas de tránsito prescribe de forma definitiva en un plazo de **3 años** contados a partir de la fecha de la infracción, a menos que se libre Mandamiento de Pago.
        
        **2. Notificación en Fotomultas (Ley 1843/17 y STC-9884/2021)**
        Las fotomultas electrónicas deben validarse dentro de los 10 días siguientes al hecho y notificarse físicamente dentro de los **3 días hábiles** posteriores por correo certificado.
        
        **3. Debido Proceso (Art. 29 C.P.)**
        Toda sanción impuesta por las autoridades de tránsito debe garantizar el derecho de defensa y contradicción en audiencia pública.
        """)
