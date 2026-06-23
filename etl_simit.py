"""
ETL Pipeline — TránsitoLegal AI
================================
Descarga datos reales del SIMIT desde datos.gov.co (API Socrata pública, sin auth),
los transforma al esquema interno de la app y los fusiona con el dataset.csv existente.

Fuente: https://www.datos.gov.co/resource/72nf-y4v3.json
Campos disponibles: vigencia, placa, fecha_multa, valor_multa, departamento, ciudad, pagado_si_no
"""

import urllib.request
import json
import pandas as pd
import numpy as np
import datetime
import os
import sys

# Forzar codificación UTF-8 para la salida en consola (Windows)
sys.stdout.reconfigure(encoding='utf-8')

# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
BASE_URL    = "https://www.datos.gov.co/resource/72nf-y4v3.json"
LIMIT       = 5000          # máx registros por descarga (sin App Token)
OFFSET_STEP = 1000          # lotes de 1000
OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), "dataset.csv")
# ───────────────────────────────────────────────────────────────────────────────


def fetch_simit_page(offset: int = 0, limit: int = 1000) -> list[dict]:
    """Descarga una página de registros SIMIT desde datos.gov.co."""
    url = f"{BASE_URL}?%24offset={offset}&%24limit={limit}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def inferir_inconsistencia(row: dict) -> str:
    """
    Regla de negocio basada en datos abiertos:
    - Si valor_multa == 0 y pagado_si_no == 'NO' → posible anulación / error de valor
    - Si la fecha tiene más de 3 años → prescripción potencial
    - Si pagado_si_no == 'NO' y vigencia < 2022 → riesgo prescripción
    - Por defecto → SIN INCONSISTENCIAS
    """
    try:
        valor = float(row.get("valor_multa", 0) or 0)
    except (ValueError, TypeError):
        valor = 0
    pagado   = str(row.get("pagado_si_no", "")).upper()
    vigencia = str(row.get("vigencia", "2024"))

    try:
        fecha = pd.to_datetime(row.get("fecha_multa", ""), dayfirst=True, errors="coerce")
        años_trans = (pd.Timestamp.now() - fecha).days / 365 if pd.notna(fecha) else 0
    except Exception:
        años_trans = 0

    if valor == 0 and pagado == "NO":
        return "ERROR VALOR: El valor registrado es $0 pero la multa figura como NO pagada. Posible nulidad."
    if años_trans >= 3 and pagado == "NO":
        return (
            f"PRESCRITO: Han transcurrido {años_trans:.1f} años. "
            "Prescripción por Art. 159 Ley 769/02. Acción de nulidad viable."
        )
    if int(vigencia) < 2022 and pagado == "NO":
        return (
            f"RIESGO PRESCRIPCION: Multa de {vigencia} sin pagar. "
            "Verificar prescripción (3 años). Revisar notificación."
        )
    if valor == 0 and pagado == "SI":
        return "ARCHIVADO: Multa de valor $0 marcada como pagada. Proceso cerrado."
    return "SIN INCONSISTENCIAS"


def inferir_clase_vehi(placa: str) -> str:
    """Infiere clase de vehículo por formato de placa colombiana."""
    placa = str(placa).upper().strip()
    if len(placa) == 6 and placa[:3].isalpha() and placa[3:].isdigit():
        return "AUTOMOVIL"
    if len(placa) == 6 and placa[:3].isalpha() and not placa[3:].isdigit():
        return "MOTOCICLETA"
    if len(placa) == 7:
        return "CAMION"
    return "AUTOMOVIL"


def inferir_cod_infraccion(row: dict) -> str:
    """Asigna código de infracción basado en valor de multa (aproximación)."""
    try:
        valor = float(row.get("valor_multa", 0) or 0)
    except (ValueError, TypeError):
        valor = 0
    if valor == 0:
        return "B01"
    if valor < 200_000:
        return "B02"
    if valor < 500_000:
        return "C02"
    if valor < 900_000:
        return "C04"
    return "D02"


def transformar_registro(row: dict, idx: int) -> dict:
    """Transforma un registro SIMIT crudo al esquema interno de la app."""
    placa   = str(row.get("placa", "SIN-PLACA")).upper().strip()
    ciudad  = row.get("ciudad", "Colombia")
    depto   = row.get("departamento", "")
    fecha   = row.get("fecha_multa", "01/01/2020")
    valor   = float(row.get("valor_multa", 0) or 0)
    pagado  = str(row.get("pagado_si_no", "NO")).upper()

    estado = "PAGADO" if pagado == "SI" else "PENDIENTE"
    inconsistencia = inferir_inconsistencia(row)
    clase  = inferir_clase_vehi(placa)
    codigo = inferir_cod_infraccion(row)

    return {
        "ID_COMPARENDO":                f"SIMIT-{row.get('vigencia', '20XX')}-{idx:05d}",
        "FECHA_INFRACCION":             fecha,
        "HORA_INFRACCION":              "00:00",
        "CC_USUARIO":                   0,          # datos abiertos no incluyen CC
        "PLACA":                        placa,
        "COD_INFRACCION":               codigo,
        "CLASE_VEHI":                   clase,
        "LUGAR_INFRACCION":             f"{ciudad}, {depto}",
        "VALOR_SANCION":                int(valor),
        "ESTADO":                       estado,
        "INCONSISTENCIA_JUSTIFICACION": inconsistencia,
    }


def descargar_dataset(total: int = LIMIT) -> pd.DataFrame:
    """
    Descarga hasta `total` registros del SIMIT en lotes y los transforma.
    Devuelve un DataFrame listo para fusionarse con dataset.csv.
    """
    registros_raw = []
    fetched = 0
    print(f"⬇️  Descargando hasta {total} registros del SIMIT (datos.gov.co)...")
    
    while fetched < total:
        lote_size = min(OFFSET_STEP, total - fetched)
        try:
            page = fetch_simit_page(offset=fetched, limit=lote_size)
        except Exception as e:
            print(f"⚠️  Error en offset {fetched}: {e}. Deteniendo.")
            break
        if not page:
            print(f"✅  No hay más registros en offset {fetched}.")
            break
        registros_raw.extend(page)
        fetched += len(page)
        print(f"   ✔ {fetched} registros descargados...")
        if len(page) < lote_size:
            break

    print(f"\n📦  Total descargado: {len(registros_raw)} registros crudos.")
    
    transformados = [
        transformar_registro(row, i)
        for i, row in enumerate(registros_raw)
    ]
    return pd.DataFrame(transformados)


def merge_con_existente(df_nuevo: pd.DataFrame, path_csv: str) -> pd.DataFrame:
    """
    Fusiona el nuevo dataset SIMIT con el dataset.csv ya existente.
    Elimina duplicados por ID_COMPARENDO.
    """
    if os.path.exists(path_csv):
        df_existente = pd.read_csv(path_csv, encoding="utf-8")
        print(f"📂  Dataset existente: {len(df_existente)} registros.")
        df_merged = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_merged.drop_duplicates(subset=["ID_COMPARENDO"], keep="last", inplace=True)
    else:
        df_merged = df_nuevo
    return df_merged


def guardar_y_reportar(df: pd.DataFrame, path_csv: str):
    """Guarda el dataset fusionado en UTF-8 y muestra estadísticas."""
    df.to_csv(path_csv, index=False, encoding="utf-8")
    print(f"\n✅  Dataset guardado en: {path_csv}")
    print(f"   Total registros: {len(df)}")
    print(f"   Con inconsistencias: {(df['INCONSISTENCIA_JUSTIFICACION'] != 'SIN INCONSISTENCIAS').sum()}")
    print(f"   Valor total multas: ${df['VALOR_SANCION'].sum():,.0f}")
    print("\n📊  Distribución de inconsistencias:")
    print(df["INCONSISTENCIA_JUSTIFICACION"].value_counts().to_string())


# ─── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TránsitoLegal AI — ETL Pipeline SIMIT Datos Abiertos")
    print("=" * 60)
    
    df_simit = descargar_dataset(total=LIMIT)
    df_final = merge_con_existente(df_simit, OUTPUT_CSV)
    guardar_y_reportar(df_final, OUTPUT_CSV)
    
    print("\n🎉 ¡Dataset listo para entrenar el modelo de IA!")
