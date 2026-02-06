## Instalación y Configuración

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión de dependencias. Para replicar el entorno exacto:

1. **Instalar uv** (si no lo tienes):
   ```bash
   curl -LsSf https://astral-sh.uv.install.sh | sh
   ```
2. **Sincronizar el proyecto:**
    Esto instalará la versión correcta de Python y todas las dependencias definidas en uv.lock:
    ```bash
    uv sync
    ```