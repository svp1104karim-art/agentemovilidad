# 🚗⚖️ TransitoLegal AI - Prototipo Web
### Repositorio: `agentemovilidad`

Un prototipo web inteligente diseñado para ayudar en la consulta, análisis y reclamación legal de comparendos de tránsito en Colombia. Esta aplicación cuenta con un panel de control interactivo, un simulador impulsado por Machine Learning y herramientas de generación de documentos legales.

---

## ✨ Características Principales
- **📊 Panel de Control (Dashboard)**: Visualiza estadísticas clave como la tasa de irregularidades, potencial de ahorro por impugnación y distribución de multas por clase de vehículo.
- **🔍 Analizador Inteligente de Comparendos**: Permite seleccionar comparendos existentes o simular nuevos y evaluarlos con un modelo de Inteligencia Artificial (Random Forest) para determinar la viabilidad de impugnación.
- **⚖️ Biblioteca Legal e Impugnación**: Acceso a fundamentos jurídicos clave sobre prescripción, caducidad y fotomultas (bajo la legislación colombiana, Ley 769 de 2002).
- **📄 Plantilla de Derecho de Petición**: Minuta lista para copiar orientada a solicitar la prescripción de comparendos con más de 3 años de antigüedad.
- **💾 Soporte UTF-8 Optimizado**: Base de datos de comparendos corregida para evitar errores de codificación con acentos y caracteres especiales.

---

## 🛠️ Pila Tecnológica
- **Lenguaje**: Python 3.14+
- **Framework Web**: Streamlit
- **Procesamiento de Datos**: Pandas
- **Machine Learning**: Scikit-Learn
- **Algoritmo**: RandomForestClassifier

---

## 📦 Instalación y Ejecución Local

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/svp1104karim-art/agentemovilidad.git
   cd agentemovilidad
   ```

2. **Instalar dependencias**:
   Asegúrate de tener Python instalado y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**:
   Usa el comando de Streamlit para lanzar el servidor local:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Estructura del Proyecto

```text
agentemovilidad/
├── app.py              # Punto de entrada de la aplicación (Lógica del Dashboard y UI)
├── dataset.csv         # Base de datos de comparendos (Codificado en UTF-8)
├── requirements.txt    # Lista de dependencias del proyecto (numpy, pandas, scikit-learn, streamlit)
└── README.md           # Este archivo de documentación
```

---

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si tienes sugerencias para mejorar el panel legal, agregar nuevas plantillas o refinar el modelo de Machine Learning, no dudes en abrir un *Issue* o enviar un *Pull Request*.

---

## 📄 Licencia
Este proyecto se distribuye bajo la Licencia MIT. Consúltala para más detalles.
