import numpy as np
import matplotlib.pyplot as plt
import torch

SEED = 42
Epochs = 5001
LR = 0.5
rng = np.random.default_rng(SEED)
Scale = 0.5 # Parametro para generar
Hidden = 2

torch.manual_seed(SEED)

X = torch.tensor([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype = torch.float32)

y = torch.tensor(
    [[0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype = torch.float32)

W1 = (torch.rand(2, Hidden) * Scale).requires_grad_()
b1 = torch.zeros(1,Hidden,requires_grad = True)

W2 = (torch.rand(Hidden, 1)*Scale).requires_grad_()
b2 = torch.zeros(1,1, requires_grad = True)

loss_history = []

for epoch in range(Epochs):

    z1 = X @ W1 + b1
    a1 = torch.sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = torch.sigmoid(z2)

    if epoch == 0:
        print(f"Predicción inicial: {y_pred} \n")


    loss = torch.mean((y-y_pred)**2)
    loss_history.append(loss.item())

    loss.backward()

# Utilizamos esta funcion de torch para no añadir estos calculos al grado computacional de backward
    with torch.no_grad():
        W1 -= LR * W1.grad
        b1 -= LR * b1.grad
        W2 -= LR * W2.grad
        b2 -= LR * b2.grad

# loss.backward hace de forma nativa acumulacion de gradiente
# Pero a nosotros no nos interesa que se sume el gradiente de la epoca anterior
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()


    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss.item(): .6f} ")


# Calculo final con gradientes actualizados

with torch.no_grad():
    z1 = X @ W1 + b1
    a1 = torch.sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = torch.sigmoid(z2)


y_class = (y_pred >= 0.5).float()
accuracy = (y_class == y).float().mean()

print("Preds:")
print(y_pred)

print("Clases predichas:")
print(y_class)

print(f"Accuracy: {accuracy.item(): .2%}")


plt.figure(figsize = (6,4))
plt.plot(loss_history)
plt.title('Curva de perdida')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.grid(True)
plt.show()
