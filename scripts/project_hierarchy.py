import os

def mostrar_jerarquia(ruta, prefijo=""):
    try:
        elementos = sorted(os.listdir(ruta))
    except PermissionError:
        return

    for i, elemento in enumerate(elementos):
        ruta_completa = os.path.join(ruta, elemento)
        es_ultimo = i == len(elementos) - 1

        if os.path.isdir(ruta_completa):
            if "__pycache__" in elemento:
                continue

            print(f"{prefijo}{'└── ' if es_ultimo else '├── '}{elemento}/")

            nuevo_prefijo = prefijo + ("    " if es_ultimo else "│   ")
            mostrar_jerarquia(ruta_completa, nuevo_prefijo)

        elif elemento.endswith(".py"):
            print(f"{prefijo}{'└── ' if es_ultimo else '├── '}{elemento}")

if __name__ == "__main__":
    ruta_proyecto = "."  # Cambia esto si tu proyecto está en otra ruta
    print("Estructura del proyecto:\n")
    mostrar_jerarquia(ruta_proyecto)