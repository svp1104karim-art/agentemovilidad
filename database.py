import sqlite3
import pandas as pd
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "comparendos.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comparendos (
            ID_COMPARENDO TEXT PRIMARY KEY,
            FECHA_INFRACCION TEXT,
            HORA_INFRACCION TEXT,
            CC_USUARIO INTEGER,
            PLACA TEXT,
            COD_INFRACCION TEXT,
            CLASE_VEHI TEXT,
            LUGAR_INFRACCION TEXT,
            VALOR_SANCION REAL,
            ESTADO TEXT,
            INCONSISTENCIA_JUSTIFICACION TEXT,
            creado_en TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def cargar_dataset_csv(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, encoding="utf-8")
    df["FECHA_INFRACCION"] = pd.to_datetime(df["FECHA_INFRACCION"], dayfirst=True)
    return df


def seed_from_csv(path_csv: str):
    conn = get_conn()
    existentes = conn.execute("SELECT COUNT(*) FROM comparendos").fetchone()[0]
    if existentes > 0:
        conn.close()
        return
    df = cargar_dataset_csv(path_csv)
    df.to_sql("comparendos", conn, if_exists="append", index=False)
    conn.close()


def obtener_todos() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM comparendos", conn, parse_dates=["FECHA_INFRACCION"])
    conn.close()
    return df


def insertar_registro(fila: dict):
    conn = get_conn()
    columnas = ", ".join(fila.keys())
    placeholders = ", ".join("?" for _ in fila)
    sql = f"INSERT OR REPLACE INTO comparendos ({columnas}) VALUES ({placeholders})"
    conn.execute(sql, list(fila.values()))
    conn.commit()
    conn.close()


def insertar_muchos(df: pd.DataFrame):
    conn = get_conn()
    df.to_sql("comparendos", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
