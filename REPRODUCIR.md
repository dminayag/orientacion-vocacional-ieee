# Reproducción del artículo

Ejecutar `python benchmark.py` con Python 3.12 para replicar la prueba local. El script sobrescribe `latencias.csv` y `rendimiento.json` con las mediciones de la nueva ejecución. El CSV incluido conserva los 10 240 pares originales en nanosegundos. El identificador de perfil es el índice, desde cero, en la lista de textos únicos de la partición 1, en orden de aparición.

Para usar el contenedor, ejecutar `docker build -t vocacional-benchmark .` y `docker run --rm vocacional-benchmark`. El Dockerfile es una configuración de reproducción. Las cifras del artículo provienen de Windows y la configuración Docker no se ejecutó en esa prueba.

Para reproducir la clasificación original, ejecutar `python -m pip install numpy==2.3.5 pandas==3.0.1 openpyxl==3.1.5` y luego `python analisis.py`. Consultar LEEME.md para la procedencia y licencia del conjunto público.
