# Results

Esta carpeta guardara resultados de experimentos.

La organizacion recomendada es una carpeta por experimento:

```text
results/
└── 01_full_finetuning/
    ├── metrics.json
    ├── training_history.json
    ├── loss_curve.png
    └── notes.md
```

No se deben generar resultados ficticios. Cada archivo debe corresponder a una ejecucion real.

## `notes.md`

Cada experimento deberia incluir un archivo `notes.md` con:

- hipotesis;
- configuracion;
- resultados;
- observaciones;
- problemas encontrados;
- conclusion;
- siguiente experimento.

Los archivos pequenos como `metrics.json`, `training_history.json`, `loss_curve.png` y `notes.md` pueden ser utiles para documentar el aprendizaje. Los checkpoints, modelos y binarios pesados deben quedar fuera del control de versiones.
