# 02_backpropagation_numpy

## Pregunta

Puede una red neuronal pequena implementada solo con NumPy aprender XOR usando una capa oculta y backpropagation manual?

## Hipotesis

Una red puramente lineal no puede resolver XOR, porque XOR no es separable con una sola frontera lineal. Si introducimos una capa oculta con activacion no lineal, la red puede transformar el espacio de entrada y aprender la relacion:

```text
0 xor 0 -> 0
1 xor 0 -> 1
0 xor 1 -> 1
1 xor 1 -> 0
```

La hipotesis concreta es:

```text
si el forward, la loss, los gradientes y las actualizaciones son correctos,
la perdida debe bajar y la accuracy final debe llegar a 1.0.
```

## Configuracion

- Archivo fuente: `notebooks/02_backpropagation_numpy.py`
- Dataset: XOR completo, 4 ejemplos.
- Entradas: `X.shape = (4, 2)`.
- Etiquetas: `y.shape = (4, 1)`.
- Neuronas ocultas: `Hidden_Neurons = 2`.
- Activacion: sigmoid en capa oculta y salida.
- Loss: MSE.
- Learning rate: `LR = 0.5`.
- Epochs: `10000`.
- Inicializacion: normal con media `0.0` y desviacion `INIT_SCALE = 0.5`.
- Seed: `42`.

## Arquitectura

La red tiene esta estructura:

```text
X -> W1,b1 -> z1 -> sigmoid -> a1 -> W2,b2 -> z2 -> sigmoid -> y_pred
```

Con shapes:

```text
X      = (4, 2)
W1     = (2, 2)
b1     = (1, 2)
z1     = (4, 2)
a1     = (4, 2)
W2     = (2, 1)
b2     = (1, 1)
z2     = (4, 1)
y_pred = (4, 1)
```

## Comprobaciones de gradientes

El script confirma que cada gradiente tiene la misma forma que su parametro:

```text
dW1 shape: (2, 2)
db1 shape: (1, 2)
dW2 shape: (2, 1)
db2 shape: (1, 1)
```

Esto es una comprobacion importante. Si `dW1` no tuviera la misma shape que `W1`, no podriamos hacer:

```python
W1 = W1 - LR * dW1
```

Lo mismo aplica a `b1`, `W2` y `b2`.

## Comprobacion de un solo paso

Antes de entrenar durante muchas epochs, el script hace una sola actualizacion.

Resultado:

```text
Loss antes de un paso:   0.2922916243810897
Loss despues de un paso: 0.28637467082417883
```

La perdida baja despues de un solo paso. Esto no demuestra que el entrenamiento entero vaya a converger, pero si indica que:

- el signo de actualizacion parece correcto;
- los gradientes tienen una direccion razonable;
- la regla `parametro = parametro - LR * gradiente` esta actuando como descenso de gradiente.

## Evolucion de la perdida

Durante el entrenamiento completo:

```text
epoch=0000 | loss=0.292292
epoch=1000 | loss=0.249931
epoch=2000 | loss=0.248929
epoch=3000 | loss=0.195270
epoch=4000 | loss=0.025647
epoch=5000 | loss=0.008377
epoch=6000 | loss=0.004754
epoch=7000 | loss=0.003267
epoch=8000 | loss=0.002472
epoch=9000 | loss=0.001981
```

Loss final:

```text
0.0016485673777804845
```

La curva no baja de forma fuerte desde el primer instante. Al principio se queda cerca de `0.25`, que es una zona tipica cuando una red binaria predice alrededor de `0.5` para todos los ejemplos. Despues de suficientes actualizaciones, la red encuentra una representacion interna util y la perdida cae con claridad.

## Predicciones finales

```text
[[0.03908334],
 [0.95662095],
 [0.95598973],
 [0.03531293]]
```

Aplicando umbral `0.5`:

```text
[[0],
 [1],
 [1],
 [0]]
```

Etiquetas reales:

```text
[[0],
 [1],
 [1],
 [0]]
```

Accuracy:

```text
1.0
```

## Interpretacion

La red aprendio XOR correctamente.

Las predicciones finales no son exactamente `0` y `1`, sino probabilidades producidas por sigmoid. Para una clasificacion binaria, una prediccion como `0.039` se interpreta como clase `0`, y una prediccion como `0.956` se interpreta como clase `1`.

Esto demuestra que la capa oculta y la activacion no lineal permitieron representar una relacion que una sola recta no puede modelar.

## Conclusion

Este experimento cierra una pieza fundamental del proyecto:

```text
forward pass -> loss -> backward pass -> update -> aprendizaje
```

La diferencia respecto a `01_red_numpy` es que aqui los parametros de la primera capa no afectan directamente a la salida. Afectan a `z1`, luego a `a1`, luego a `z2`, luego a `y_pred`, y solo al final a la loss. Backpropagation permite calcular esa cadena de dependencias de forma sistematica.

## Siguiente paso

Antes de pasar a PyTorch, conviene estudiar `math_breakdown.md` y poder explicar:

- que representa `z1`;
- que representa `a1`;
- que representa `z2`;
- que representa `dz2`;
- por que `dW2 = a1.T @ dz2`;
- por que `da1 = dz2 @ W2.T`;
- por que `dW1 = X.T @ dz1`;
- por que los bias usan suma sobre `axis=0`.
