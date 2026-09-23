import numpy as np
import matplotlib.pyplot as plt

SEED = 42
Hidden_Neurons = 2
LR = 0.5
Epochs = 10000
INIT_SCALE = 0.5
rng = np.random.default_rng(SEED)

X = np.array([
    (0,0),
    (1,0),
    (0,1),
    (1,1)
], dtype = float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype = float)


print("X shape:", X.shape)
print("y shape:", y.shape)

# Inicializamos pesos de forma aleatoria en una matriz 2 x Hidden_Neurons
W1 = rng.normal(loc = 0.0, scale = INIT_SCALE, size= (2, Hidden_Neurons)) 
print(W1)
b1 = np.zeros((1, Hidden_Neurons))

W2 = rng.normal(loc = 0.0, scale = INIT_SCALE, size= (Hidden_Neurons,1))
b2 = np.zeros((1,1))

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)


def sigmoid (x):
    return 1/(1+ np.exp(-x))

def forward(X, W1, b1, W2, b2):

    z1 = X @ W1 + b1 # Combinación lineal de la neurona
    a1 = sigmoid(z1) # Activación

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "y_pred": y_pred,
    }

    return y_pred, cache


y_pred, cache = forward(X, W1, b1, W2, b2)

print("y real:")
print(y)

print("y_pred inicial:")
print(y_pred)

print("y_pred shape:", y_pred.shape)

def MSE(y_pred,y):
    error = y_pred - y
    return np.mean(error**2)

loss = MSE(y_pred, y)
print("Loss inicial:", loss)

def Derivada_Activacion(a):
    # da/dz = a* (1-a)
    return a * (1-a)


def backward(X, y, W2, cache):
    """
    Función destinada al cálculo de los gradientes de la red.

    Devolvemos:
        dw1, db1, dw2, db2
    """

    n = X.shape[0]

    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    y_pred = cache["y_pred"]

    # Gradiente de la perdida con respecto a la predicción

    error = y_pred - y
    dloss_dypred = 2.0 * error/n

    # Gradiente respecto a z2

    dypred_dz2 = Derivada_Activacion(y_pred)
    dz2 = dloss_dypred * dypred_dz2

    # Gradientes de W2 y b2

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis = 0, keepdims = True)

    # ========================================================
    # 4. Propagar el error hacia la capa oculta
    # ========================================================

    da1 = dz2 @ W2.T

    # ========================================================
    # 5. Gradiente respecto a z1
    # ========================================================

    da1_dz1 = Derivada_Activacion(a1)
    dz1 = da1 * da1_dz1

    # ========================================================
    # 6. Gradientes de W1 y b1
    # ========================================================

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


loss_history = []


# ============================================================
# Training loop
# ============================================================

for epoch in range(Epochs):
    y_pred, cache = forward(X, W1, b1, W2, b2)
    loss = MSE(y_pred, y)

    dW1, db1, dW2, db2 = backward(X, y, W2, cache)

    W1 = W1 - LR * dW1
    b1 = b1 - LR * db1
    W2 = W2 - LR * dW2
    b2 = b2 - LR * db2

    loss_history.append(float(loss))

    if epoch % 1000 == 0:
        print(f"epoch={epoch:04d} | loss={loss:.6f}")


# ============================================================
# Evaluacion
# ============================================================

y_pred_final, _ = forward(X, W1, b1, W2, b2)
y_class = (y_pred_final >= 0.5).astype(int)
accuracy = np.mean(y_class == y)

print("Predicciones finales:")
print(y_pred_final)

print("Clases finales:")
print(y_class)

print("Etiquetas reales:")
print(y.astype(int))

print("Accuracy:", accuracy)
print("Loss final:", loss_history[-1])


# ============================================================
# Curva de perdida
# ============================================================

plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("Curva de perdida - XOR con NumPy")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()
