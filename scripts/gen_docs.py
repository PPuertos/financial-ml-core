import os
from pathlib import Path

def generate_docs():
    # 1. Configuración de rutas
    source_package = "finml_core"  # La carpeta con tu código
    output_dir = Path("docs/reference") # Donde queremos los archivos .md

    print(f"🚀 Iniciando generación de docs en: {output_dir}")

    for root, dirs, files in os.walk(source_package):
        for file in files:
            # Solo procesamos archivos .py y saltamos los __init__.py
            if file.endswith(".py") and not file.startswith("__"):
                
                # RUTA DEL MÓDULO (para el ::: de mkdocstrings)
                # Ejemplo: finml_core/metrics/statistics.py -> finml_core.metrics.statistics
                file_path = Path(root) / file
                module_path = str(file_path.with_suffix("")).replace(os.sep, ".")
                
                # RUTA DEL ARCHIVO .MD
                # Ejemplo: docs/reference/metrics/statistics.md
                # Quitamos "finml_core" de la ruta del archivo para que no sea repetitivo
                relative_path = Path(root).relative_to(source_package)
                target_file = output_dir / relative_path / f"{file.replace('.py', '.md')}"
                
                # Crear la carpeta de destino
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Escribir el contenido
                with open(target_file, "w", encoding="utf-8") as f:
                    title = file.replace(".py", "").replace("_", " ").title()
                    f.write(f"# {title}\n\n")
                    f.write(f"::: {module_path}\n")
                
                print(f"✅ Generado: {target_file}")

if __name__ == "__main__":
    generate_docs()