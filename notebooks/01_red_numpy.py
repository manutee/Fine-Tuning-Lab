import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
"""
El objetivo de esta prueba inicial es 
definir y comprender los conceptos elementales
de las capas de neuronas: perdida, derivadas, descenso de gradiente, etcétera.

Pregunta inicial:
Puede una red minima aprender una relación sencilla 
tipo: ŷ  = wx + b (donde ŷ es la prediccion, x la entrada y b el sesgo) ?
Comenzamos con una regresión lineal simple, solo Numpy
"""


# Parametros de configuracion iniciales

SAMPLES = 100

TRUE_W = 3.0
TRUE_B = 2.0
NOISE = 0.5

LR = 0.05
NUM_EPOCHS = 100
rng = np.random.default_rng(42)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

RESULTS_DIR = PROJECT_ROOT / "results" / "01_red_numpy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Creamos los datos sinteticos 

x = np.linspace(-2.0, 2.0, SAMPLES)
noise = rng.normal(loc=0.0, scale=NOISE, size=SAMPLES)

y = TRUE_W * x + TRUE_B + noise

# Visualizamos los datos
plt.figure(figsize = (6,4))
plt.scatter(x,y, alpha = 0.9)
plt.title("Datos sinteticos generados")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig(RESULTS_DIR / "data_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# Inicializamos los parametros de entrenamiento
w = rng.normal()
b = rng.normal()

print("w inicial:", w)
print("b inicial:", b)

# Funciones de predicción, error y gradientes

def predict (x,w,b):
    return w*x + b

def mse_loss(y_pred, y_true):
    error = y_pred - y_true
    return np.mean(error**2)

def compute_gradientes (x, y_true, y_pred):
    error = y_pred - y_true

    dw = np.mean(2.0*error*x)
    db = np.mean(2.0*error)
    return dw, db

# ============================================================
# Un solo paso de actualizacion
# ============================================================
"""
w_step = w
b_step = b

y_pred_before = predict(x, w_step, b_step)
loss_before = mse_loss(y_pred_before, y)

dw, db = compute_gradientes(x, y, y_pred_before)

w_step = w_step - LR * dw
b_step = b_step - LR * db

y_pred_after = predict(x, w_step, b_step)
loss_after = mse_loss(y_pred_after, y)

print("Loss antes:", loss_before)
print("Loss despues:", loss_after)
print("dw:", dw)
print("db:", db)
print("w despues de un paso:", w_step)
print("b despues de un paso:", b_step)

"""


# Reiniciamos parámetros y damos paso al training loop

w = rng.normal()
b = rng.normal()
loss_history =[]
training_history = []

for epoch in range(NUM_EPOCHS):

    y_pred = predict(x,w,b)
    loss = mse_loss (y_pred, y)

    dw, db = compute_gradientes(x,y,y_pred)

    w = w - LR * dw
    b = b - LR * db

    loss_history.append(float(loss))
    training_history.append(
        {
            "epoch": int(epoch),
            "loss": float(loss),
            "dw": float(dw),
            "db": float(db),
            "w_after_update": float(w),
            "b_after_update": float(b),
        }
    )

    if epoch % 20 == 0:
        print(
            f"epoch={epoch:03d} | "
            f"loss={loss:.6f} | "
            f"w={w:.4f} | "
            f"b={b:.4f}"
        )


print("TRUE_W:", TRUE_W)
print("TRUE_B:", TRUE_B)

print("w aprendido:", w)
print("b aprendido:", b)

print("error en w:", abs(TRUE_W - w))
print("error en b:", abs(TRUE_B - b))

# Curva de perdida

plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("Curva de perdida")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# Recta aprendida
# ============================================================

final_pred = predict(x, w, b)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, alpha=0.8, label="Datos reales")
plt.plot(x, final_pred, color="red", linewidth=2, label="Modelo aprendido")
plt.title("Datos reales vs modelo aprendido")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "learned_line.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# Registro del experimento
# ============================================================

loss_array = np.array(loss_history)
loss_initial = float(loss_array[0])
loss_final = float(loss_array[-1])
best_epoch = int(np.argmin(loss_array))
best_loss = float(loss_array[best_epoch])

error_w = float(abs(TRUE_W - w))
error_b = float(abs(TRUE_B - b))
finite_losses = bool(np.isfinite(loss_array).all())
loss_decreased = bool(loss_final < loss_initial)

metrics = {
    "experiment": "01_red_numpy",
    "question": "Puede una red minima con NumPy aprender una relacion lineal y = wx + b?",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "config": {
        "samples": SAMPLES,
        "true_w": TRUE_W,
        "true_b": TRUE_B,
        "noise": NOISE,
        "learning_rate": LR,
        "num_epochs": NUM_EPOCHS,
        "seed": 42,
    },
    "results": {
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "learned_w": float(w),
        "learned_b": float(b),
        "abs_error_w": error_w,
        "abs_error_b": error_b,
    },
    "checks": {
        "finite_losses": finite_losses,
        "loss_decreased": loss_decreased,
    },
}

history = {
    "experiment": "01_red_numpy",
    "loss_history": loss_history,
    "training_history": training_history,
}

notes = f"""# 01_red_numpy

## Pregunta

Puede una red minima implementada solo con NumPy aprender una relacion lineal del tipo `y = wx + b`?

## Hipotesis

Si el forward pass, la funcion de perdida, los gradientes y la actualizacion estan bien calculados, la perdida deberia bajar y los parametros aprendidos deberian acercarse a TRUE_W y TRUE_B.

## Configuracion

- Samples: {SAMPLES}
- TRUE_W: {TRUE_W}
- TRUE_B: {TRUE_B}
- Noise: {NOISE}
- Learning rate: {LR}
- Epochs: {NUM_EPOCHS}
- Seed: 42

## Resultados

- Loss inicial: {loss_initial:.6f}
- Loss final: {loss_final:.6f}
- Mejor loss: {best_loss:.6f}
- Epoch de mejor loss: {best_epoch}
- w aprendido: {float(w):.6f}
- b aprendido: {float(b):.6f}
- Error absoluto en w: {error_w:.6f}
- Error absoluto en b: {error_b:.6f}

## Observaciones

- La perdida {'bajo' if loss_decreased else 'no bajo'} durante el entrenamiento.
- Las perdidas {'son finitas' if finite_losses else 'contienen valores no finitos'}.
- El modelo aprende una recta que debe compararse visualmente con la nube de puntos en `learned_line.png`.
- Como hay ruido en los datos, no se espera que la perdida llegue exactamente a cero.

## Conclusion

Este experimento demuestra el ciclo minimo de entrenamiento: forward pass, calculo de perdida, gradientes manuales y actualizacion de parametros mediante descenso de gradiente.

## Siguiente experimento

Probar una red pequena con capa oculta y backpropagation manual.
"""

(RESULTS_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)
(RESULTS_DIR / "training_history.json").write_text(
    json.dumps(history, indent=2),
    encoding="utf-8",
)
(RESULTS_DIR / "notes.md").write_text(notes, encoding="utf-8")

print("Resultados guardados en:", RESULTS_DIR)
print("Archivos generados:")
print("-", RESULTS_DIR / "metrics.json")
print("-", RESULTS_DIR / "training_history.json")
print("-", RESULTS_DIR / "data_scatter.png")
print("-", RESULTS_DIR / "loss_curve.png")
print("-", RESULTS_DIR / "learned_line.png")
print("-", RESULTS_DIR / "notes.md")
