"""Funciones reutilizables para entrenamiento.

Responsabilidad:
    Guardar piezas comunes de entrenamiento solo cuando se repitan entre
    experimentos y no oculten el razonamiento educativo.

No hay training loop implementado aqui. El forward pass, la perdida, el backward
pass, el optimizer step, la validacion, el checkpointing, mixed precision y la
acumulacion de gradientes se escribiran progresivamente.

Posibles funciones futuras:
    - fijar seeds;
    - contar parametros entrenables;
    - congelar o descongelar capas;
    - guardar y restaurar checkpoints simples;
    - registrar el historial de perdida y metricas.
"""
