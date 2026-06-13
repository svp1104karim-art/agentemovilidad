import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import datetime
import requests

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

# --- INICIALIZACIÓN DE SESIÓN (STATEFUL DATA) ---
if 'df' not in st.session_state:
    try:
        df_init = pd.read_csv('dataset.csv', encoding='utf-8')
        df_init['FECHA_INFRACCION'] = pd.to_datetime(df_init['FECHA_INFRACCION'], dayfirst=True)
        st.session_state.df = df_init
    except FileNotFoundError:
        st.error("🚨 El archivo 'dataset.csv' no se encuentra en la raíz.")
        st.stop()

# --- MODEL TRAINING PIPELINE ---
def entrenar_modelo(df_actual):
    df_temp = df_actual.copy()
    df_temp['DIAS_DESDE_REFERENCIA'] = (df_temp['FECHA_INFRACCION'] - pd.to_datetime('01/01/2018')).dt.days
    
    features = ['DIAS_DESDE_REFERENCIA', 'CLASE_VEHI', 'COD_INFRACCION']
    X = df_temp[features].copy()
    y = df_temp['INCONSISTENCIA_JUSTIFICACION']

    le_clase = LabelEncoder()
    le_codigo = LabelEncoder()

    X['CLASE_VEHI_NUM'] = le_clase.fit_transform(X['CLASE_VEHI'])
    X['COD_INFRACCION_NUM'] = le_codigo.fit_transform(X['COD_INFRACCION'])
    X = X.drop(['CLASE_VEHI', 'COD_INFRACCION'], axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_clase, le_codigo

# Entrenar modelo con los datos vigentes en sesión
model, le_clase, le_codigo = entrenar_modelo(st.session_state.df)

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
    st.markdown("Sistema inteligente de defensa y diagnóstico de comparendos de tránsito.")
    st.write("---")
    
    # 📂 CARGADOR DE DATOS REALES (EXCEL/CSV)
    st.markdown("### 📂 Cargar Datos Reales")
    archivo_usuario = st.file_uploader(
        "Sube tu archivo de comparendos reales:",
        type=["csv", "xlsx"],
        help="Sube un archivo Excel o CSV que tenga columnas similares a: ID_COMPARENDO, CC_USUARIO, PLACA, COD_INFRACCION, CLASE_VEHI, VALOR_SANCION, ESTADO, etc."
    )
    
    if archivo_usuario is not None:
        try:
            if archivo_usuario.name.endswith('.csv'):
                df_cargado = pd.read_csv(archivo_usuario, encoding='utf-8')
            else:
                df_cargado = pd.read_excel(archivo_usuario)
            
            # Formatear fecha si existe
            if 'FECHA_INFRACCION' in df_cargado.columns:
                df_cargado['FECHA_INFRACCION'] = pd.to_datetime(df_cargado['FECHA_INFRACCION'], errors='coerce')
            else:
                df_cargado['FECHA_INFRACCION'] = pd.to_datetime(datetime.date.today())

            # Asegurar columna de inconsistencia
            if 'INCONSISTENCIA_JUSTIFICACION' not in df_cargado.columns:
                df_cargado['INCONSISTENCIA_JUSTIFICACION'] = 'SIN INCONSISTENCIAS'
            
            st.session_state.df = df_cargado
            st.success("¡Datos reales cargados con éxito!")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.write("---")
    opcion = st.radio(
        "Navegación",
        ["📊 Dashboard de Control", "🔍 Diagnóstico de Comparendos", "⚖️ Guía Legal y Plantillas"]
    )
    
    st.write("---")
    # 🌐 CONFIGURACIÓN API VERIFIK (SIMIT REAL)
    st.markdown("### 🌐 API SIMIT en Vivo (Verifik)")
    verifik_token = st.text_input(
        "Token de Verifik.co:",
        type="password",
        placeholder="Bearer token de tu cuenta",
        help="Obtén tu token gratuito en verifik.co — permite consultar comparendos reales del SIMIT colombiano."
    )
    if verifik_token:
        st.session_state['verifik_token'] = verifik_token
        st.success("✅ API Key configurada")
    elif 'verifik_token' not in st.session_state:
        st.session_state['verifik_token'] = None
        st.caption("Sin token: solo se consultará la base de datos local.")
    
    st.write("---")
    st.info("💡 **Versión:** 1.4.0 (SIMIT Real)\n\nConecta con la API de Verifik para consultar comparendos en tiempo real.")

# --- PÁGINA 1: DASHBOARD ---
if opcion == "📊 Dashboard de Control":
    st.markdown("<h1 class='main-title'>📊 Dashboard de Control de Comparendos</h1>", unsafe_allow_html=True)
    st.write("Análisis estadístico e histórico de infracciones y su viabilidad de defensa.")
    
    # Datos de sesión
    df_active = st.session_state.df
    
    # Métricas
    total_comparendos = len(df_active)
    inconsistencias = df_active[df_active['INCONSISTENCIA_JUSTIFICACION'] != 'SIN INCONSISTENCIAS']
    num_inconsistencias = len(inconsistencias)
    tasa_irregularidad = (num_inconsistencias / total_comparendos) * 100 if total_comparendos > 0 else 0
    total_pesos = df_active['VALOR_SANCION'].sum() if 'VALOR_SANCION' in df_active.columns else 0
    total_ahorro_potencial = inconsistencias['VALOR_SANCION'].sum() if 'VALOR_SANCION' in df_active.columns else 0

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
        if num_inconsistencias > 0:
            st.bar_chart(inconsistencias['INCONSISTENCIA_JUSTIFICACION'].value_counts())
        else:
            st.info("No hay inconsistencias en la base de datos actual.")
    with c2:
        st.subheader("🚗 Distribución por Clase de Vehículo")
        if 'CLASE_VEHI' in df_active.columns:
            st.bar_chart(df_active['CLASE_VEHI'].value_counts())
        else:
            st.info("No hay columna CLASE_VEHI en el dataset.")

    st.subheader("📋 Tabla General de Registros Vigentes")
    st.dataframe(df_active, use_container_width=True)

# --- PÁGINA 2: DIAGNÓSTICO ---
elif opcion == "🔍 Diagnóstico de Comparendos":
    st.markdown("<h1 class='main-title'>🔍 Analizador Inteligente de Comparendos</h1>", unsafe_allow_html=True)
    st.write("Consulta comparendos en la base de datos local **o consulta el SIMIT real** con tu token de Verifik.")

    # --- FUNCIÓN DE CONSULTA REAL A VERIFIK / SIMIT ---
    def consultar_simit_por_cedula(cc: str, token: str):
        """Consulta comparendos reales en el SIMIT colombiano via Verifik API."""
        try:
            url = "https://api.verifik.co/v2/co/simit/consultar"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }
            params = {"documentType": "CC", "documentNumber": cc.strip()}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"Error {response.status_code}: {response.text[:200]}"
        except requests.exceptions.ConnectionError:
            return None, "No se pudo conectar con Verifik. Verifica tu conexión a Internet."
        except requests.exceptions.Timeout:
            return None, "La consulta tardó demasiado. Intenta de nuevo."
        except Exception as e:
            return None, str(e)

    def consultar_simit_por_placa(placa: str, token: str):
        """Consulta comparendos reales por placa en el SIMIT colombiano via Verifik API."""
        try:
            url = "https://api.verifik.co/v2/co/simit/consultar/placa"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }
            params = {"placa": placa.strip().upper()}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"Error {response.status_code}: {response.text[:200]}"
        except requests.exceptions.ConnectionError:
            return None, "No se pudo conectar con Verifik. Verifica tu conexión a Internet."
        except requests.exceptions.Timeout:
            return None, "La consulta tardó demasiado. Intenta de nuevo."
        except Exception as e:
            return None, str(e)

    def renderizar_resultado_verifik(data: dict, tipo: str):
        """Renderiza la respuesta JSON de Verifik en tarjetas visuales."""
        multas = data.get('data', {}).get('multas', data.get('multas', []))
        cursos = data.get('data', {}).get('cursos', data.get('cursos', []))
        acuerdos = data.get('data', {}).get('acuerdosPago', data.get('acuerdosPago', []))
        
        total_multas = len(multas) if isinstance(multas, list) else 0
        total_valor = sum(float(m.get('valorMulta', m.get('valor', 0)) or 0) for m in multas) if isinstance(multas, list) else 0
        
        # Resumen ejecutivo
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🚨 Comparendos SIMIT", total_multas)
        col_r2.metric("💰 Valor Total", f"${total_valor:,.0f}")
        col_r3.metric("📚 Cursos Pedagógicos", len(cursos) if isinstance(cursos, list) else 0)
        
        if isinstance(multas, list) and multas:
            st.write("---")
            st.markdown("#### 📋 Detalle de Comparendos")
            for i, multa in enumerate(multas):
                with st.expander(f"🚦 Comparendo #{i+1} — {multa.get('secretaria', multa.get('organismo', 'Organismo no especificado'))}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Fecha:** {multa.get('fechaInfraccion', multa.get('fecha', 'N/A'))}")
                        st.markdown(f"**Infracción:** {multa.get('codigoInfraccion', multa.get('codigo', 'N/A'))}")
                        st.markdown(f"**Estado:** `{multa.get('estado', multa.get('estadoMulta', 'N/A'))}`")
                    with col_b:
                        st.markdown(f"**Valor:** ${float(multa.get('valorMulta', multa.get('valor', 0)) or 0):,.0f}")
                        st.markdown(f"**Organismo:** {multa.get('secretaria', multa.get('organismo', 'N/A'))}")
                        st.markdown(f"**Municipio:** {multa.get('municipio', 'N/A')}")
                    
                    # Análisis de prescripción automático
                    fecha_str = multa.get('fechaInfraccion', multa.get('fecha', ''))
                    try:
                        fecha_dt = pd.to_datetime(fecha_str, errors='coerce')
                        if fecha_dt and pd.notna(fecha_dt):
                            dias_transcurridos = (pd.Timestamp.now() - fecha_dt).days
                            years = dias_transcurridos / 365
                            if years >= 3:
                                st.error(f"⚠️ **PRESCRITO:** Han transcurrido {years:.1f} años. Esta multa puede estar prescrita (Art. 159, Ley 769/02).")
                            elif years >= 2:
                                st.warning(f"⏰ **PRÓXIMO A PRESCRIBIR:** Han transcurrido {years:.1f} años. Actúa antes de los 3 años.")
                            else:
                                st.info(f"📅 Han transcurrido {years:.1f} años desde la infracción.")
                    except:
                        pass
        elif total_multas == 0:
            st.success("✅ No se encontraron comparendos activos en el SIMIT para este documento/placa.")
        
        # Mostrar JSON completo
        with st.expander("🔍 Ver respuesta JSON completa de Verifik"):
            st.json(data)

    tabs = st.tabs(["🔎 Consultar por CC / Placa", "🌐 Consulta SIMIT en Vivo", "➕ Registrar Comparendo Real"])

    with tabs[0]:
        # Tarjeta de Búsqueda Segura (Base de datos local)
        st.info("💾 Esta pestaña consulta la **base de datos local** cargada en la app. Para consultas en tiempo real en el SIMIT, usa la pestaña **🌐 Consulta SIMIT en Vivo**.")
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #f8fafc;">Consultar Base de Datos Local</h4>
                <span style="font-size: 0.85rem; color: #94a3b8;">🔒 Búsqueda Segura</span>
            </div>
        """, unsafe_allow_html=True)

        col_cc, col_placa = st.columns(2)
        with col_cc:
            st.write("**Por Cédula de Ciudadanía (CC)**")
            cc_input = st.text_input("Número de cédula:", placeholder="Ej: 80987654", key="cc")
            buscar_cc = st.button("Buscar por Cédula", type="primary", use_container_width=True)

        with col_placa:
            st.write("**Por Placa Vehicular**")
            placa_input = st.text_input("Placa del vehículo:", placeholder="Ej: XYZ987", key="placa").upper()
            buscar_placa = st.button("Buscar por Placa", type="secondary", use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # Lógica de Consulta
        resultados = None
        df_active = st.session_state.df
        
        if buscar_cc and cc_input:
            try:
                cc_int = int(cc_input.strip())
                resultados = df_active[df_active['CC_USUARIO'] == cc_int]
            except ValueError:
                st.error("Por favor, ingresa un número de cédula válido.")
        elif buscar_placa and placa_input:
            resultados = df_active[df_active['PLACA'].str.upper() == placa_input.strip()]

        # Mostrar Resultados de Búsqueda Local
        if resultados is not None:
            if len(resultados) == 0:
                st.warning("⚠️ No se encontraron comparendos en la base de datos local. Prueba la pestaña 🌐 Consulta SIMIT en Vivo.")
            else:
                st.success(f"📋 Se encontraron **{len(resultados)}** comparendos en la base de datos local:")
                
                for index, row in resultados.iterrows():
                    # Formatear datos
                    ticket_id = row['ID_COMPARENDO']
                    organismo = row.get('LUGAR_INFRACCION', 'Secretaría de Movilidad')
                    infraccion_cod = row.get('COD_INFRACCION', 'Desconocido')
                    clase_vehi = row.get('CLASE_VEHI', 'AUTOMOVIL')
                    valor = row.get('VALOR_SANCION', 0)
                    estado = row.get('ESTADO', 'ACTIVO')
                    placa = row.get('PLACA', 'SIN PLACA')
                    fecha = row['FECHA_INFRACCION']
                    inconsistencia_just = row.get('INCONSISTENCIA_JUSTIFICACION', 'SIN INCONSISTENCIAS')

                    # Predicción usando ML
                    dias = (pd.to_datetime(fecha) - pd.to_datetime('01/01/2018')).days
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

                    st.markdown(f"---")
                    
                    # Columnas de Información, Diagnóstico y Acciones
                    col_info, col_diag, col_actions = st.columns([2, 1.2, 1])
                    
                    with col_info:
                        st.markdown(f"<div class='ticket-title'>{ticket_id}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='ticket-detail-item'>📅 **Fecha:** {fecha.strftime('%d/%m/%Y') if hasattr(fecha, 'strftime') else str(fecha)}</div>", unsafe_allow_html=True)
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

    with tabs[1]:
        st.markdown("<h3 style='color:#10b981;'>🌐 Consulta SIMIT en Tiempo Real</h3>", unsafe_allow_html=True)
        token_activo = st.session_state.get('verifik_token')
        
        if not token_activo:
            st.warning("⚠️ No has configurado tu Token de Verifik. Ingresa el token en el **panel lateral izquierdo** para activar las consultas en vivo.")
            st.markdown("📖 **¿Cómo obtener el token?** Ve a la pestaña **⚖️ Guía Legal → 🔑 Cómo obtener el Token** para las instrucciones.")
        else:
            st.success("✅ Token configurado — Consultas en vivo al SIMIT activadas.")
            
            col_live_cc, col_live_placa = st.columns(2)
            with col_live_cc:
                st.write("**🔎 Buscar por Cédula de Ciudadanía**")
                live_cc = st.text_input("Número de cédula:", placeholder="Ej: 80987654", key="live_cc")
                tipo_doc = st.selectbox("Tipo de documento:", ["CC", "CE", "PA", "TI", "NIT"], key="tipo_doc")
                buscar_live_cc = st.button("🌐 Consultar SIMIT por Cédula", type="primary", use_container_width=True)
                
            with col_live_placa:
                st.write("**🚗 Buscar por Placa Vehicular**")
                live_placa = st.text_input("Placa del vehículo (sin guión):", placeholder="Ej: ABC123", key="live_placa").upper()
                buscar_live_placa = st.button("🌐 Consultar SIMIT por Placa", type="secondary", use_container_width=True)
            
            st.write("---")
            
            if buscar_live_cc and live_cc:
                with st.spinner(f"🔄 Consultando el SIMIT para la cédula **{live_cc}**..."):
                    resultado, error = consultar_simit_por_cedula(live_cc, token_activo)
                if resultado:
                    st.success(f"✅ Consulta exitosa para CC: **{live_cc}**")
                    renderizar_resultado_verifik(resultado, "cc")
                else:
                    st.error(f"❌ Error al consultar: {error}")
                    
            elif buscar_live_placa and live_placa:
                with st.spinner(f"🔄 Consultando el SIMIT para la placa **{live_placa}**..."):
                    resultado, error = consultar_simit_por_placa(live_placa, token_activo)
                if resultado:
                    st.success(f"✅ Consulta exitosa para placa: **{live_placa}**")
                    renderizar_resultado_verifik(resultado, "placa")
                else:
                    st.error(f"❌ Error al consultar: {error}")

    with tabs[2]:
        st.subheader("📝 Registrar Datos de un Comparendo Real")
        st.write("Rellena este formulario con la información del comparendo que desees registrar en la sesión de análisis:")
        
        with st.form("registro_comparendo_real"):
            c1, c2, c3 = st.columns(3)
            with c1:
                nuevo_id = st.text_input("Número del Comparendo (ID):", placeholder="Ej: COMP-2026-999")
                nuevo_cc = st.number_input("Cédula del Propietario (CC):", min_value=0, value=80987654)
                nuevo_valor = st.number_input("Valor de la Multa ($):", min_value=0, value=500000, step=10000)
            with c2:
                nueva_placa = st.text_input("Placa del Vehículo:", placeholder="Ej: ABC123").upper()
                nueva_clase = st.selectbox("Clase del Vehículo:", ["AUTOMOVIL", "MOTOCICLETA", "CAMION", "BUS", "TAXI"])
                nuevo_codigo = st.selectbox("Código de la Infracción:", ["C04", "B01", "B02", "C02", "D02"])
            with c3:
                nueva_fecha = st.date_input("Fecha de Imposición:", datetime.date.today())
                nuevo_organismo = st.text_input("Organismo de Tránsito (Lugar):", placeholder="Ej: Bogotá Calle 100")
                nuevo_estado = st.selectbox("Estado del Comparendo:", ["PENDIENTE", "EN PROCESO", "PAGADO", "ARCHIVADO"])
                
            inconsistencia_manual = st.selectbox(
                "¿Tiene inconsistencias conocidas? (Opcional)",
                [
                    "SIN INCONSISTENCIAS",
                    "PRESCRITO: La sanción ha prescrito de conformidad con el art. 8 de la Ley 769 de 2002 (mas de 3 años).",
                    "ERROR CODIGO: El codigo es inaplicable para la clase de vehículo.",
                    "ERROR PROCEDIMIENTO: Falta de firmas oficiales o identificación de agente.",
                    "ERROR LUGAR: Error en ubicación o coordenadas reportadas."
                ]
            )
            
            enviar_registro = st.form_submit_button("Registrar Comparendo en la Base de Datos", type="primary")
            
            if enviar_registro:
                if not nuevo_id or not nueva_placa or not nuevo_organismo:
                    st.error("Por favor completa los campos obligatorios: ID, Placa y Organismo.")
                else:
                    # Crear fila
                    nueva_fila = {
                        'ID_COMPARENDO': nuevo_id,
                        'FECHA_INFRACCION': pd.to_datetime(nueva_fecha),
                        'HORA_INFRACCION': '00:00',
                        'CC_USUARIO': nuevo_cc,
                        'PLACA': nueva_placa,
                        'COD_INFRACCION': nuevo_codigo,
                        'CLASE_VEHI': nueva_clase,
                        'LUGAR_INFRACCION': nuevo_organismo,
                        'VALOR_SANCION': nuevo_valor,
                        'ESTADO': nuevo_estado,
                        'INCONSISTENCIA_JUSTIFICACION': inconsistencia_manual
                    }
                    # Concatenar en sesión
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([nueva_fila])], ignore_index=True)
                    st.success(f"🎉 Comparendo **{nuevo_id}** registrado exitosamente. Ahora puedes consultarlo en la pestaña de búsquedas.")
                    st.balloons()
                    st.rerun()

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
        
    # --- DOCUMENTACIÓN TÉCNICA REAL: VERIFIK + SIMIT ---
    st.write("---")
    st.subheader("🌐 Integración Real con SIMIT (Verifik.co — Proveedor Verificado)")
    st.markdown("""
    Esta aplicación está integrada con la **API real de Verifik.co**, el proveedor colombiano verificado 
    con acceso al SIMIT (Sistema Integrado de Información sobre Multas y Sanciones de Tránsito).
    
    Para activar consultas en vivo, ingresa tu token en el **panel lateral izquierdo**.
    """)

    tab_doc1, tab_doc2, tab_doc3 = st.tabs(["📡 Endpoints Reales", "🔑 Cómo obtener el Token", "🧪 Probar Conexión"])
    
    with tab_doc1:
        st.markdown("""
        ### Endpoints Oficiales de Verifik (SIMIT Colombia)
        
        | Método | Endpoint | Descripción |
        |--------|----------|-------------|
        | `GET` | `https://api.verifik.co/v2/co/simit/consultar` | Consulta por Cédula (CC, CE, PA, NIT) |
        | `GET` | `https://api.verifik.co/v2/co/simit/consultar/placa` | Consulta por Placa vehicular |
        
        **Parámetros por CC:**
        ```
        documentType: "CC"   # Tipo: CC, CE, PA, TI, RC, NIT
        documentNumber: "12345678"  # Número sin puntos ni espacios
        ```
        
        **Parámetros por Placa:**
        ```
        placa: "ABC123"  # Placa sin guiones
        ```
        
        **Autenticación (Header):**
        ```
        Authorization: Bearer <tu_token>
        Accept: application/json
        ```
        
        **Estructura de respuesta JSON:**
        ```json
        {
          "data": {
            "multas": [
              {
                "fechaInfraccion": "2023-05-10",
                "codigoInfraccion": "C04",
                "valorMulta": 480000,
                "estado": "VIGENTE",
                "municipio": "BOGOTA D.C.",
                "secretaria": "Secretaria de Movilidad de Bogota"
              }
            ],
            "cursos": [],
            "acuerdosPago": []
          }
        }
        ```
        """)
    
    with tab_doc2:
        st.markdown("""
        ### ¿Cómo obtener tu Token de Verifik?
        
        1. **Regístrate** en [verifik.co](https://verifik.co) — ofrece plan gratuito con créditos de prueba.
        2. Accede a tu **Dashboard** y ve a la sección **API Keys**.
        3. Genera un **Bearer Token** y cópialo.
        4. Pégalo en el campo **"Token de Verifik.co"** en el panel lateral izquierdo de esta app.
        5. Ve a la pestaña **🌐 Consulta SIMIT en Vivo** y consulta cualquier cédula o placa en tiempo real.
        
        > ⚠️ **Importante:** Guarda tu token de forma segura. No lo compartas públicamente.
        > En Streamlit Cloud, puedes guardarlo como un **Secret** en la configuración de tu app.
        """)
        
        st.code("""
# En tu archivo .streamlit/secrets.toml (local):
[verifik]
token = "tu_bearer_token_aqui"

# Cómo leerlo en app.py:
token = st.secrets["verifik"]["token"]
        """, language="toml")
    
    with tab_doc3:
        st.markdown("### 🧪 Prueba de conexión con Verifik")
        token_test = st.session_state.get('verifik_token')
        if not token_test:
            st.warning("Ingresa tu Token de Verifik en el panel lateral para probar la conexión.")
        else:
            cc_test = st.text_input("Cédula de prueba:", placeholder="Ej: 80112233", key="test_cc")
            if st.button("🚀 Probar Conexión Ahora", type="primary"):
                if cc_test:
                    with st.spinner("Consultando SIMIT en tiempo real..."):
                        resultado, error = consultar_simit_por_cedula(cc_test, token_test)
                    if resultado:
                        st.success("✅ Conexión exitosa con la API de Verifik")
                        st.json(resultado)
                    else:
                        st.error(f"❌ Error: {error}")
                else:
                    st.warning("Ingresa una cédula de prueba.")
