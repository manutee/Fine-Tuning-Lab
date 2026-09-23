# Experiments

Esta carpeta contendra scripts completos de experimentos reproducibles.

Cada experimento debe responder a una pregunta concreta. La logica principal debe poder leerse de arriba abajo, con los hiperparametros visibles y sin esconder pasos importantes detras de abstracciones prematuras.

## Estructura recomendada

1. Pregunta o hipotesis.
2. Configuracion visible.
3. Carga de datos.
4. Carga del modelo.
5. Preparacion de la estrategia.
6. Entrenamiento.
7. Evaluacion.
8. Guardado de resultados.
9. Conclusiones.

## Esquema conceptual

```python
"""
Pregunta:
    Que quiero comprobar en este experimento?
"""

# 1. Configuracion visible
# 2. Preparar datos
# 3. Preparar modelo
# 4. Definir estrategia de entrenamiento
# 5. Ejecutar entrenamiento paso a paso
# 6. Evaluar
# 7. Guardar metricas, historial y notas
# 8. Escribir conclusion
```

Este esquema no es un training loop funcional. Los detalles se implementaran progresivamente durante el aprendizaje.

## Posibles experimentos futuros

```text
01_full_finetuning.py
02_head_only.py
03_last_layers.py
04_lora_manual.py
05_lora_peft.py
```

No hay experimentos implementados todavia.
