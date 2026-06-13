#!/usr/bin/env python3
"""
Script para eliminar el directorio sospechoso: transito_ai_prototipo.py
Uso: python3 cleanup.py
"""

import os
import shutil
import subprocess
import sys

def remove_suspicious_files():
    """Elimina el directorio sospechoso y hace commit"""
    
    directory = "transito_ai_prototipo.py"
    
    print(f"🔍 Buscando directorio: {directory}")
    
    if os.path.exists(directory):
        print(f"⚠️  Directorio encontrado. Eliminando...")
        try:
            shutil.rmtree(directory)
            print(f"✅ Directorio {directory} eliminado exitosamente")
        except Exception as e:
            print(f"❌ Error al eliminar: {e}")
            return False
    else:
        print(f"ℹ️  Directorio {directory} no encontrado")
        return True
    
    # Configurar git
    print("\n📝 Configurando Git...")
    subprocess.run(["git", "config", "user.email", "cleanup@script.local"], check=False)
    subprocess.run(["git", "config", "user.name", "Cleanup Script"], check=False)
    
    # Agregar cambios
    print("📦 Agregando cambios a Git...")
    subprocess.run(["git", "add", "-A"], check=False)
    
    # Hacer commit
    print("💾 Haciendo commit...")
    result = subprocess.run(
        ["git", "commit", "-m", "Remove suspicious directory: transito_ai_prototipo.py"],
        check=False
    )
    
    if result.returncode == 0:
        print("✅ Commit realizado exitosamente")
        
        # Push
        print("🚀 Empujando cambios...")
        push_result = subprocess.run(["git", "push", "origin", "main"], check=False)
        
        if push_result.returncode == 0:
            print("✅ Cambios empujados a main exitosamente")
            return True
        else:
            print("❌ Error al empujar cambios")
            return False
    else:
        print("ℹ️  No hay cambios para hacer commit")
        return True

if __name__ == "__main__":
    print("=" * 50)
    print("🧹 Script de Limpieza - Archivo Sospechoso")
    print("=" * 50)
    
    success = remove_suspicious_files()
    
    print("=" * 50)
    if success:
        print("✅ Limpieza completada exitosamente")
        sys.exit(0)
    else:
        print("❌ Limpieza fallida")
        sys.exit(1)