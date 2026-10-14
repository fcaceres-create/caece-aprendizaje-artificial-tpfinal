# Despliegue: frontend + API en la web

## Arquitectura

```
Navegador ──► Contenedor Docker (app/Dockerfile)
               └─ FastAPI (uvicorn)
                   ├─ /                 frontend propio (app/web: HTML + CSS + JS, sin dependencias)
                   ├─ /health, /model/info, /predict, /predict/batch   API REST
                   └─ /docs             documentación interactiva de la API
```

- **Un solo contenedor y una sola URL.** Como el frontend se sirve desde el mismo origen que la API, no
  hay problemas de CORS.
- **CORS habilitado igual.** Está activo por si se quiere alojar el frontend aparte, por ejemplo en GitHub
  Pages: alcanza con poner la URL de la API en `app/web/config.js`.
- **Streamlit sigue disponible** como herramienta interna en el `docker-compose`. No se publica.

## Probar localmente con Docker

```bash
docker compose -f app/docker-compose.yml up --build -d
# Frontend:   http://localhost:8000          (agregar ?demo para cargar ejemplos automáticamente)
# API docs:   http://localhost:8000/docs
# Streamlit:  http://localhost:8501
docker compose -f app/docker-compose.yml down   # para detenerlo
```

Verificado: imagen de 1,9 GB; la API usa unos 175 MB de RAM.

## Publicar en Render (recomendado)

1. Crear una cuenta en https://render.com. Se puede ingresar con la cuenta de GitHub.
2. Ir a **New → Blueprint** y elegir el repositorio `caece-aprendizaje-artificial-tpfinal`. Render lee
   `render.yaml` y crea el servicio `captacion-clientes` con el plan gratuito.
3. Esperar el primer build, que tarda unos 5 a 10 minutos. La URL queda con la forma
   `https://captacion-clientes-XXXX.onrender.com`.
4. Cada `git push` a `main` vuelve a desplegar el servicio automáticamente.

**Limitación del plan gratuito:** el servicio se duerme tras 15 minutos sin uso, y la primera visita
después tarda alrededor de un minuto en despertarlo. Antes de una presentación conviene abrir la URL unos
minutos antes.

## Alternativa: Hugging Face Spaces

Es gratis, tiene 16 GB de RAM y se duerme tras 48 horas sin uso. Requiere un repositorio aparte en Hugging
Face:

1. Crear un Space de tipo **Docker** (plantilla *Blank*).
2. Subir ahí el contenido del repo con un `Dockerfile` en la raíz, que puede ser una copia de
   `app/Dockerfile`.
3. Agregar al principio del `README.md` del Space este encabezado:

   ```yaml
   ---
   title: Captación de clientes
   sdk: docker
   app_port: 7860
   ---
   ```

4. Definir la variable `PORT=7860` en la configuración del Space. El contenedor usa `$PORT`.
