import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 300
LR = 0.1
SAMPLES = 700
NOISE = 3
Train_split = 0.7
Batch_size = 32

RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "04_clasificacion_pytorch_dataloader"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

samples_per_class = SAMPLES // 2

# Centros de los datasets sinteticos alrededor de (-1,-1) y (1,1) respectivamente
Negs_center = torch.tensor([-1.0, -1.0])
Pos_center = torch.tensor([1.0, 1.0])

# El modulo torch randn genera valores aleatorios con distribucuion normal centrada en 0
# Creamos una matriz 200 x 2 donde el ruido controla la dispersion del dataset
# Y la suma del centro de negativos desplaza todo el grupo alrededor del punto (-1,-1)
Negs_X = torch.randn(samples_per_class, 2) * NOISE + Negs_center
Pos_X = torch.randn(samples_per_class, 2) * NOISE + Pos_center

# Vector de etiquetas binarias
Negs_Y = torch.zeros(samples_per_class, 1)
Pos_Y = torch.ones(samples_per_class,1)

# El modulo cat une las entradas por sus filas, entonces torch.cat([(200 x 2), (200 x 2)]) = (400 x 2)
# Primeros 200 ejemplos a clase negativa y siguientes 200 a clase positiva
X = torch.cat([Negs_X, Pos_X], dim = 0)
Y = torch.cat([Negs_Y, Pos_Y], dim = 0)

# Aplicamos una permutacion a los ejemplos para mezclar positivos y negativos
permutation = torch.randperm(SAMPLES)

X = X[permutation]
Y = Y[permutation]

# Convertimos la matriz (SAMPLES X 1) en un vector 1D con las etiquetas
labels = Y.squeeze(1)
negs = labels == 0
pos = labels == 1

plt.figure(figsize=(8,6))

plt.scatter(X[negs, 0], X[negs,1], label = 'Clase negativa', alpha = 0.8)
plt.scatter(X[pos, 0], X[pos,1], label = 'Clase positiva', alpha = 0.8)

plt.grid(True)
plt.title('Dataset Sintetico mezclado')
plt.xlabel('Característica postiva')
plt.ylabel('Característica negativa')
plt.legend()
plt.axis("equal")
plt.savefig(RESULTS_DIR / "dataset_by_class.png", dpi=150, bbox_inches="tight")
plt.close()


train_size = int(SAMPLES * Train_split)

# Aclaracion: train_size = 280
# si hacemos [:train_size] -> coge todos los valores desde el inicio hasta 280
# de esta forma hacemos el split efectivo
X_train = X[:train_size]
Y_train = Y[:train_size]

X_val = X[train_size:]
Y_val = Y[train_size:]

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader( train_dataset, batch_size = Batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = Batch_size, shuffle = False)

# Declarando que nuestra clase es un tipo de nn.Module permitimos que use metodos de esta clase base y que pytorch lo interprete como un modelo
class BinaryLinearClassifier(nn.Module):
    def __init__(self):
        super().__init__() # Con super.init inicializamos la parte heredada del modulo padre        
        self.linear = nn.Linear( in_features = 2, out_features = 1,)

    def forward (self, x): # x es el batch entregado por DataLoader
        logits = self.linear(x)# Aqui entregamos el batch a la capa lineal
        return logits


X_batch, y_batch = next(iter(train_loader))
model = BinaryLinearClassifier()
logits = model(X_batch)


# Utilizamos entropía cruzada binaria como funcion de peridda
# Con logits significa que la funcion espera recibir la salida de la capa lineal
# El clasico z = w1*x1 + w2 ...
loss_Func = nn.BCEWithLogitsLoss()
# La funcion aplica internamente sigmoid 
# Y usa la ecuacion de BCE: loss_i = -[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
 
# Stochastic Gradient Descent
# Aplica el clasico  w <- w - LR * d(loss)/dw
optimizer = torch.optim.SGD(model.parameters(), lr = LR)

# La funcion compara los logits con las etiquetas 
loss = loss_Func(logits, y_batch)


val_loss_history = []
train_loss_history =[]
val_acc_treshold_history = []
val_base_acc_history = []
coverage_history = []



for epoch in range(EPOCHS):
    # Por convención se emplea model.train para poner la capa lineal en modo entrenamiento
    # Actualmente no cambia el resultado, pero en un futuro lo hará
    model.train()

    train_sum = 0.0 # Suma ponderada de perdidas
    train_examples = 0 # Total de ejemplos procesados

    for X_batch, y_batch in train_loader:
        # Entrenamiento del modelo recorriendo los batches
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_Func(logits, y_batch)
        loss.backward()
        optimizer.step()

        # Telemetría de perdida de entrenamiento
        # Hacemos esto y no batch_size directamente porque el ultimo batch tendrá 24 ejemplos, no 32
        curr_batch_size = X_batch.shape[0]

        # Ponderamos la perdida de todos los ejemplos del batch 
        # Asi podemos calcular de forma precisa la train_loss por epoch
        train_sum += curr_batch_size * loss.item()
        train_examples += curr_batch_size

    train_loss = train_sum/train_examples
    train_loss_history.append(train_loss)

    # De nuevo ahora lo hacemos por convención
    model.eval()

    val_sum = 0.0
    val_examples = 0
    val_correct_preds = 0
    val_correct_pos_preds = 0
    val_correct_neg_preds = 0
    confident_corr = 0
    confident_examples = 0

    with torch.no_grad():
            # Calculamos metricas en el split de validacion
            for X_batch, y_batch in val_loader:

                logits = model(X_batch)
                loss = loss_Func(logits, y_batch)

                probs = torch.sigmoid(logits)

                #================================================================
                # Accuracy con umbral base
                #===============================================================

                Base_threshold = (probs>= 0.5).float()
                val_correct_preds += (Base_threshold == y_batch).sum().item()


                #================================================================
                # Accuracy con incertidumbre
                #===============================================================

                Positive_threshold = (probs >= 0.55)
                Negative_threshold = (probs <= 0.45)

                # Hacemos un OR, confianza puede ser satisfecha por cualquier umbral
                Confidence = Positive_threshold | Negative_threshold
                # Convertimos en float si se satisface el umbral positivo
                preds = Positive_threshold.float()
                correct_preds = preds == y_batch

                # Con el AND lógico hacemos la poda de los ejemplos que se encuentren en incertidumbre
                confident_corr += (correct_preds & Confidence).sum().item()
                confident_examples += Confidence.sum().item()

                curr_batch_size = X_batch.shape[0]
                val_sum += loss.item() * curr_batch_size
                val_examples += curr_batch_size


    # Accuracy con umbral base 0.5

    base_accuracy = val_correct_preds/val_examples

    # Accuracy calibrada con incertidumbre
    confidence_accuracy = confident_corr / confident_examples
    coverage = confident_examples / val_examples

    val_loss = val_sum/val_examples

    val_loss_history.append(val_loss)
    val_base_acc_history.append (base_accuracy)
    val_acc_treshold_history.append(confidence_accuracy)
    coverage_history.append(coverage)

    if epoch % 10 == 0 or epoch == (EPOCHS -1):
         print(
              f"Epoch {epoch+1:03d}/{EPOCHS}|"
              f"train loss:{train_loss:.4f}|"
              f"val loss:{val_loss:.4f}|"
              f"val base accuracy: {base_accuracy: .2%}|"
              f"val accuracy threshold: {confidence_accuracy: .2%}|"
              f"coverage threshold: {coverage: .2%}"
         )


# Inidcamos model.eval para usar el modelo en inferencia post entrenamiento
model.eval()
# Sin el margen el grafico se cortaría exactamente en los puntos mas alejado por arriba y por abajo
margin = 0.5

# X tiene shape (Samples x 2), cogemos todas las filas y la primera característica
# De la primera característica cogemos la menor y la mayor para los limites horizontales
x_min = X[:, 0].min().item() - margin
x_max = X[:, 0].max().item() + margin

# Cogemos la segunda caracterísitca y hacemos lo mismo para delimitar los verticales
y_min = X[:, 1].min().item() - margin
y_max = X[:, 1].max().item() + margin

# Construimos una cuadricula de 250 x 250 puntos
# Ambos grid_x y grid_y tienen forma (250, 250) donde cada posicion (i,j) es un punto X o Y en el plano
grid_x, grid_y = np.meshgrid(np.linspace(x_min,x_max, 250),np.linspace(y_min,y_max,250),)

# Con ravel aplanamos la matriz a 1D y con column stack contruimos una matriz donde
# Se corresponde cada valor de x con su valor de y, es decir (250 x 250) -> (62500, 2)
grid_points = torch.tensor(np.column_stack([grid_x.ravel(), grid_y.ravel()]), dtype = torch.float32,)
# Luego convertimos la matriz en un tensor compatible con los pesos

# Solo queremos generar predicciones para dibujar, por ende hacemos no_grad
with torch.no_grad():
    # Le pasamos al modelo los grid points con shape (62500, 2)    
    grid_logits = model(grid_points)
    # Los logits toman shape (62500, 1)
    grid_probs = torch.sigmoid(grid_logits)
    # Hacemos un reshape donde cada probabilidad ocupa de nuevo su posicion dentro de la malla
    grid_probs = grid_probs.reshape(grid_x.shape).numpy()
    # El .numpy convierte el tensor en una matriz numpy

labels = Y.squeeze(1)
negative_mask = labels == 0
positive_mask = labels == 1

plt.figure(figsize=(8, 7))

# Significa filled contours, que son regiones rellenas de color
# Grid_probs son las probabilidades que da el modelo en ese punto,
# Y levels la cantidad de bandas de color donde se distribuyen las probs
background = plt.contourf(grid_x,grid_y,grid_probs,levels=20,cmap="RdBu_r",alpha=0.35,)

# Escala que explica cada color de fondo
plt.colorbar(background, label="Probabilidad de clase positiva")

plt.contour(grid_x,grid_y,grid_probs,levels=[0.3, 0.5, 0.7],colors="black",linestyles=["--", "-", "--"],)

plt.scatter(X[negative_mask, 0],X[negative_mask, 1],label="Clase 0",alpha=0.7,)

plt.scatter(X[positive_mask, 0],X[positive_mask, 1],label="Clase 1",alpha=0.7,)

plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
plt.title("Frontera de decision y regiones de confianza")
plt.grid(True)
plt.legend()
plt.axis("equal")

plt.savefig(RESULTS_DIR / "decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()



epochs_axis = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs_axis, train_loss_history, label="Train loss")
plt.plot(epochs_axis, val_loss_history, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("Curvas de perdida")
plt.grid(True)
plt.legend()

plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(epochs_axis, val_base_acc_history)
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Accuracy de validacion base")
plt.grid(True)
plt.ylim(0.0, 1.0)

plt.savefig(RESULTS_DIR / "base_accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(epochs_axis, val_acc_treshold_history, label="Confidence accuracy")
plt.plot(epochs_axis, coverage_history, label="Coverage")
plt.xlabel("Epoch")
plt.ylabel("Metric value")
plt.title("Metricas de confianza en validacion")
plt.grid(True)
plt.ylim(0.0, 1.0)
plt.legend()

plt.savefig(RESULTS_DIR / "confidence_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

metrics = {
    "experiment": "04_clasificacion_pytorch_dataloader",
    "configuration": {
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "samples": SAMPLES,
        "noise_std": NOISE,
        "train_fraction": Train_split,
        "batch_size": Batch_size,
        "base_threshold": 0.5,
        "negative_confidence_threshold": 0.3,
        "positive_confidence_threshold": 0.7,
    },
    "dataset": {
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
    },
    "model": {
        "name": "BinaryLinearClassifier",
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    },
    "final_metrics": {
        "train_loss": train_loss_history[-1],
        "validation_loss": val_loss_history[-1],
        "base_accuracy": val_base_acc_history[-1],
        "confidence_accuracy": val_acc_treshold_history[-1],
        "coverage": coverage_history[-1],
    },
}

training_history = {
    "epoch": list(epochs_axis),
    "train_loss": train_loss_history,
    "validation_loss": val_loss_history,
    "base_accuracy": val_base_acc_history,
    "confidence_accuracy": val_acc_treshold_history,
    "coverage": coverage_history,
}

(RESULTS_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print(f"Resultados guardados en: {RESULTS_DIR}")
