# Desglose matematico de `02_backpropagation_numpy`

Este documento explica con detalle la matematica del experimento XOR implementado en:

```text
notebooks/02_backpropagation_numpy.py
```

El objetivo no es memorizar formulas, sino entender que significa cada variable, por que cada gradiente tiene esa forma y como se relacionan las variables durante forward y backward.

## 1. Problema que estamos resolviendo

El dataset XOR es:

```text
x1  x2  y
0   0   0
1   0   1
0   1   1
1   1   0
```

En codigo:

```python
X = np.array([
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0],
], dtype=float)
```

Las shapes son:

```text
X.shape = (4, 2)
y.shape = (4, 1)
```

Esto significa:

```text
4 ejemplos
2 caracteristicas por ejemplo
1 etiqueta por ejemplo
```

Cada fila de `X` es un ejemplo. Cada fila de `y` es la respuesta correcta de ese ejemplo.

## 2. Por que XOR necesita una capa oculta

Una neurona lineal simple calcula algo del tipo:

```text
z = x1*w1 + x2*w2 + b
```

Luego puede aplicar una activacion para producir una clase. Pero si solo tenemos una frontera lineal, la red separa el plano con una recta.

XOR no puede separarse con una unica recta:

```text
(0,0) -> 0
(1,1) -> 0
(1,0) -> 1
(0,1) -> 1
```

Los puntos de clase `1` estan en esquinas opuestas, y los puntos de clase `0` estan en las otras dos esquinas. Una sola linea no puede dejar unos a un lado y otros al otro.

Por eso usamos:

```text
entrada -> capa oculta -> activacion no lineal -> salida
```

La capa oculta crea una representacion intermedia. La activacion no lineal permite que esa representacion no sea simplemente otra transformacion lineal.

## 3. Arquitectura del experimento

El modelo es:

```text
z1 = X @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)
```

Con `Hidden_Neurons = 2`, las shapes son:

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

La red tiene dos capas entrenables:

```text
W1, b1 -> parametros de la capa oculta
W2, b2 -> parametros de la capa de salida
```

## 4. Que significa una combinacion lineal

Una combinacion lineal es una suma ponderada.

Para una neurona:

```text

z = x1*w1 + x2*w2 + b
```

Los pesos deciden cuanto influye cada entrada. El bias desplaza el resultado.

En forma matricial hacemos muchas combinaciones lineales a la vez.

```python
z1 = X @ W1 + b1
```

Con dos neuronas ocultas, `W1` tiene forma `(2, 2)`:

```text
W1 =
[
  [w_x1_h1, w_x1_h2],
  [w_x2_h1, w_x2_h2],
]
```

Para un ejemplo `i` y una neurona oculta `h`:

```text
z1[i,h] = X[i,0]*W1[0,h] + X[i,1]*W1[1,h] + b1[0,h]
```

Esto significa:

```text
la neurona oculta h mira las dos entradas del ejemplo i,
las pondera con sus pesos,
suma su bias,
y produce un valor bruto z1[i,h].
```

El valor `z1` se llama "preactivacion" porque todavia no paso por sigmoid.

## 5. Que hace la activacion sigmoid

La sigmoid es:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

La usamos asi:

```python
a1 = sigmoid(z1)
```

Convierte cada valor de `z1` en un valor entre `0` y `1`.

Interpretacion:

```text
z muy negativo -> sigmoid(z) cerca de 0
z = 0          -> sigmoid(z) = 0.5
z muy positivo -> sigmoid(z) cerca de 1
```

La activacion no lineal es necesaria porque:

```text
lineal -> lineal
```

sigue siendo lineal, pero:

```text
lineal -> sigmoid -> lineal
```

ya puede representar relaciones no lineales.

## 6. Segunda combinacion lineal

Despues de la capa oculta:

```python
z2 = a1 @ W2 + b2
```

Con `Hidden_Neurons = 2`, `a1` tiene dos valores por ejemplo:

```text
a1[i,0] -> activacion de la neurona oculta 1 para el ejemplo i
a1[i,1] -> activacion de la neurona oculta 2 para el ejemplo i
```

`W2` tiene forma `(2, 1)`:

```text
W2 =
[
  [w_h1_out],
  [w_h2_out],
]
```

Para un ejemplo `i`:

```text
z2[i,0] = a1[i,0]*W2[0,0] + a1[i,1]*W2[1,0] + b2[0,0]
```

Es decir:

```text
la salida mira las activaciones ocultas,
las pondera,
suma un bias,
y produce un valor bruto z2.
```

Luego:

```python
y_pred = sigmoid(z2)
```

`y_pred` es la prediccion final, tambien entre `0` y `1`.

## 7. Funcion de perdida MSE

La loss usada es:

```python
loss = mean((y_pred - y) ** 2)
```

Matematicamente:

```text
L = (1/n) * sum_i (y_pred_i - y_i)^2
```

Donde:

```text
n = numero de ejemplos = 4
```

La loss mide cuanto se equivoca el modelo.

Si `y_pred_i` esta cerca de `y_i`, el error es pequeno. Si esta lejos, el error al cuadrado es grande.

## 8. Objetivo del backward pass

Durante el forward calculamos:

```text
X -> z1 -> a1 -> z2 -> y_pred -> loss
```

Durante el backward queremos calcular:

```text
dL/dW1
dL/db1
dL/dW2
dL/db2
```

Cada gradiente responde:

```text
si cambio este parametro un poquito, cuanto cambia la loss?
```

Ejemplo:

```text
dL/dW2[0,0]
```

significa:

```text
si aumento un poquito el peso W2[0,0], la loss sube o baja?
y cuanto?
```

Si el gradiente es positivo, aumentar el parametro aumentaria la loss. Para reducir la loss, lo bajamos.

Si el gradiente es negativo, aumentar el parametro bajaria la loss. Al restar un negativo, lo subimos.

La actualizacion siempre tiene esta forma:

```text
parametro = parametro - LR * gradiente
```

## 9. Grafo de dependencias

Las variables dependen unas de otras:

```text
W1,b1 -> z1 -> a1 -> z2 -> y_pred -> L
W2,b2 --------^
```

Mas explicitamente:

```text
L depende de y_pred
y_pred depende de z2
z2 depende de a1, W2 y b2
a1 depende de z1
z1 depende de X, W1 y b1
```

`X` e `y` son datos. No se entrenan.

`W1`, `b1`, `W2`, `b2` son parametros. Si se entrenan.

Backpropagation aplica la regla de la cadena desde la loss hacia atras.

## 10. Regla de la cadena

La regla de la cadena dice:

```text
si L depende de a,
y a depende de z,
entonces:

dL/dz = dL/da * da/dz
```

En nuestro caso:

```text
L -> y_pred -> z2
```

Entonces:

```text
dL/dz2 = dL/dy_pred * dy_pred/dz2
```

Y para llegar mas atras:

```text
L -> y_pred -> z2 -> a1 -> z1 -> W1
```

Por eso cada paso del backward multiplica gradientes locales.

## 11. Primer gradiente: `error = y_pred - y`

En codigo:

```python
error = y_pred - y
```

Para cada ejemplo:

```text
error_i = y_pred_i - y_i
```

Interpretacion:

```text
error_i > 0 -> la red predijo demasiado alto
error_i < 0 -> la red predijo demasiado bajo
error_i = 0 -> la red acerto
```

Shape:

```text
error = (4, 1)
```

Una fila por ejemplo.

## 12. Gradiente de la loss respecto a la prediccion

Codigo:

```python
dloss_dypred = 2.0 * error / n
```

La loss es:

```text
L = (1/n) * sum_i (y_pred_i - y_i)^2
```

Para un ejemplo:

```text
dL/dy_pred_i = (2/n) * (y_pred_i - y_i)
```

Por eso:

```text
dL/dy_pred = 2 * error / n
```

Significado:

```text
dloss_dypred dice como cambia la loss si cambia la prediccion final.
```

Si una prediccion esta demasiado alta, este gradiente sera positivo. Si esta demasiado baja, sera negativo.

Shape:

```text
dloss_dypred = (4, 1)
```

## 13. Derivada de sigmoid

La sigmoid es:

```text
a = 1 / (1 + exp(-z))
```

Su derivada es:

```text
da/dz = a * (1 - a)
```

En codigo:

```python
def Derivada_Activacion(a):
    return a * (1 - a)
```

Esta funcion recibe `a`, no `z`.

Para la salida:

```python
dypred_dz2 = Derivada_Activacion(y_pred)
```

Esto significa:

```text
dy_pred/dz2
```

Es decir:

```text
cuanto cambia la prediccion final si cambia z2?
```

Si `y_pred` esta cerca de `0.5`, la sigmoid cambia bastante. Si `y_pred` esta cerca de `0` o `1`, cambia poco porque esta saturada.

## 14. Gradiente respecto a `z2`

Codigo:

```python
dz2 = dloss_dypred * dypred_dz2
```

Matematicamente:

```text
dL/dz2 = dL/dy_pred * dy_pred/dz2
```

Esto es regla de la cadena.

`dz2` significa:

```text
cuanto cambia la loss si cambia z2?
```

Shape:

```text
dz2 = (4, 1)
```

Hay un valor por ejemplo.

`dz2` es una senal de error en la capa de salida antes de la activacion.

## 15. Gradiente de `W2`

Forward:

```python
z2 = a1 @ W2 + b2
```

Para un ejemplo `i`:

```text
z2[i,0] = a1[i,0]*W2[0,0] + a1[i,1]*W2[1,0] + b2[0,0]
```

Queremos:

```text
dL/dW2
```

Para el peso `W2[0,0]`:

```text
dz2[i,0]/dW2[0,0] = a1[i,0]
```

Porque `W2[0,0]` aparece multiplicando a `a1[i,0]`.

Por regla de la cadena:

```text
dL/dW2[0,0] = sum_i dL/dz2[i,0] * dz2[i,0]/dW2[0,0]
```

Sustituyendo:

```text
dL/dW2[0,0] = sum_i dz2[i,0] * a1[i,0]
```

Para el segundo peso:

```text
dL/dW2[1,0] = sum_i dz2[i,0] * a1[i,1]
```

En forma matricial:

```python
dW2 = a1.T @ dz2
```

Shapes:

```text
a1.shape   = (4, 2)
a1.T.shape = (2, 4)
dz2.shape  = (4, 1)

a1.T @ dz2 = (2, 1)
```

Y:

```text
W2.shape = (2, 1)
```

El gradiente tiene la misma shape que el parametro.

Interpretacion:

```text
dW2 mide como deberian cambiar los pesos que conectan la capa oculta
con la salida.
```

Si una neurona oculta estuvo muy activa en ejemplos donde la salida fue demasiado alta, su peso hacia la salida recibira una correccion hacia abajo.

Si estuvo muy activa en ejemplos donde la salida fue demasiado baja, su peso recibira una correccion hacia arriba.

## 16. Gradiente de `b2`

Forward:

```text
z2[i,0] = ... + b2[0,0]
```

El bias `b2` se suma directamente a cada ejemplo.

Entonces:

```text
dz2[i,0]/db2[0,0] = 1
```

Por regla de la cadena:

```text
dL/db2[0,0] = sum_i dL/dz2[i,0] * dz2[i,0]/db2[0,0]
```

Como la segunda parte vale `1`:

```text
dL/db2[0,0] = sum_i dz2[i,0]
```

Codigo:

```python
db2 = np.sum(dz2, axis=0, keepdims=True)
```

`axis=0` suma verticalmente, sobre los ejemplos.

`keepdims=True` mantiene la shape:

```text
db2.shape = (1, 1)
```

Igual que:

```text
b2.shape = (1, 1)
```

Interpretacion:

```text
db2 dice si conviene desplazar hacia arriba o hacia abajo
la preactivacion final z2 para todos los ejemplos.
```

## 17. Propagar el error hacia la capa oculta: `da1`

Codigo:

```python
da1 = dz2 @ W2.T
```

Queremos saber:

```text
dL/da1
```

Es decir:

```text
cuanto cambia la loss si cambia la activacion oculta a1?
```

Recordemos:

```text
z2[i,0] = a1[i,0]*W2[0,0] + a1[i,1]*W2[1,0] + b2[0,0]
```

Para una activacion oculta:

```text
dz2[i,0]/da1[i,0] = W2[0,0]
dz2[i,0]/da1[i,1] = W2[1,0]
```

Por regla de la cadena:

```text
dL/da1[i,h] = dL/dz2[i,0] * dz2[i,0]/da1[i,h]
```

Sustituyendo:

```text
dL/da1[i,h] = dz2[i,0] * W2[h,0]
```

En forma matricial:

```python
da1 = dz2 @ W2.T
```

Shapes:

```text
dz2.shape  = (4, 1)
W2.T.shape = (1, 2)

dz2 @ W2.T = (4, 2)
```

Y:

```text
a1.shape = (4, 2)
```

Interpretacion:

```text
da1 reparte la senal de error de la salida hacia las neuronas ocultas.
```

Si una conexion `W2[h,0]` es grande, la neurona oculta `h` influye mucho en la salida. Por eso recibe mas senal de error.

Si una conexion `W2[h,0]` es pequena, esa neurona influye menos en la salida. Por eso recibe menos senal de error.

## 18. Gradiente respecto a `z1`

Codigo:

```python
da1_dz1 = Derivada_Activacion(a1)
dz1 = da1 * da1_dz1
```

Sabemos:

```text
a1 = sigmoid(z1)
```

Queremos:

```text
dL/dz1
```

Por regla de la cadena:

```text
dL/dz1 = dL/da1 * da1/dz1
```

Donde:

```text
da1/dz1 = a1 * (1 - a1)
```

Por eso:

```text
dz1 = da1 * a1 * (1 - a1)
```

Shape:

```text
dz1 = (4, 2)
```

Interpretacion:

```text
dz1 es la senal de error en la capa oculta antes de la activacion.
```

Esto ya no es simplemente el error de salida. Es el error de salida propagado hacia atras, filtrado por:

- los pesos `W2`, que dicen cuanto influyo cada neurona oculta;
- la derivada de sigmoid, que dice si esa neurona podia cambiar mucho o estaba saturada.

## 19. Gradiente de `W1`

Forward:

```python
z1 = X @ W1 + b1
```

Para un ejemplo `i` y neurona oculta `h`:

```text
z1[i,h] = X[i,0]*W1[0,h] + X[i,1]*W1[1,h] + b1[0,h]
```

Queremos:

```text
dL/dW1
```

Para un peso concreto:

```text
dz1[i,h]/dW1[j,h] = X[i,j]
```

Porque `W1[j,h]` aparece multiplicando a la entrada `X[i,j]`.

Por regla de la cadena:

```text
dL/dW1[j,h] = sum_i dL/dz1[i,h] * dz1[i,h]/dW1[j,h]
```

Sustituyendo:

```text
dL/dW1[j,h] = sum_i dz1[i,h] * X[i,j]
```

En forma matricial:

```python
dW1 = X.T @ dz1
```

Shapes:

```text
X.shape    = (4, 2)
X.T.shape  = (2, 4)
dz1.shape  = (4, 2)

X.T @ dz1  = (2, 2)
```

Y:

```text
W1.shape = (2, 2)
```

Interpretacion:

```text
dW1 dice como cambiar los pesos que conectan las entradas con las neuronas ocultas.
```

Este gradiente es mas indirecto que `dW2`, porque `W1` no afecta directamente a la salida. Afecta a:

```text
W1 -> z1 -> a1 -> z2 -> y_pred -> loss
```

Backpropagation calcula exactamente esa cadena.

## 20. Gradiente de `b1`

Forward:

```text
z1[i,h] = ... + b1[0,h]
```

Entonces:

```text
dz1[i,h]/db1[0,h] = 1
```

Por regla de la cadena:

```text
dL/db1[0,h] = sum_i dL/dz1[i,h]
```

Codigo:

```python
db1 = np.sum(dz1, axis=0, keepdims=True)
```

Shape:

```text
db1.shape = (1, 2)
```

Igual que:

```text
b1.shape = (1, 2)
```

Interpretacion:

```text
db1 ajusta el umbral de activacion de cada neurona oculta.
```

Si una neurona oculta necesita activarse mas para ciertos ejemplos, el bias puede desplazarse. Si necesita activarse menos, tambien.

## 21. Resumen completo del backward

Codigo:

```python
error = y_pred - y
dloss_dypred = 2.0 * error / n

dypred_dz2 = Derivada_Activacion(y_pred)
dz2 = dloss_dypred * dypred_dz2

dW2 = a1.T @ dz2
db2 = np.sum(dz2, axis=0, keepdims=True)

da1 = dz2 @ W2.T

da1_dz1 = Derivada_Activacion(a1)
dz1 = da1 * da1_dz1

dW1 = X.T @ dz1
db1 = np.sum(dz1, axis=0, keepdims=True)
```

Lectura conceptual:

```text
1. error:
   cuanto se equivoco la red en la salida.

2. dloss_dypred:
   cuanto cambia la loss si cambia y_pred.

3. dypred_dz2:
   cuanto cambia y_pred si cambia z2.

4. dz2:
   cuanto cambia la loss si cambia z2.

5. dW2:
   como cambian los pesos de salida para bajar la loss.

6. db2:
   como cambia el bias de salida para bajar la loss.

7. da1:
   como se reparte el error hacia las activaciones ocultas.

8. dz1:
   cuanto cambia la loss si cambia z1.

9. dW1:
   como cambian los pesos de la primera capa para bajar la loss.

10. db1:
    como cambian los bias de la primera capa para bajar la loss.
```

## 22. Por que usamos transpuestas

Aparecen dos transpuestas:

```python
dW2 = a1.T @ dz2
dW1 = X.T @ dz1
```

No son trucos. Son la forma matricial de sumar contribuciones de todos los ejemplos.

Para `dW2`:

```text
a1.T = (hidden, ejemplos)
dz2  = (ejemplos, salida)

a1.T @ dz2 = (hidden, salida)
```

Esto agrega, para cada neurona oculta, cuanto contribuyo a los errores de salida de todos los ejemplos.

Para `dW1`:

```text
X.T  = (features, ejemplos)
dz1  = (ejemplos, hidden)

X.T @ dz1 = (features, hidden)
```

Esto agrega, para cada entrada y cada neurona oculta, cuanto contribuyo esa entrada al error interno de esa neurona.

## 23. Por que los bias se suman

Los pesos conectan una entrada concreta con una neurona concreta.

Los bias se suman a todos los ejemplos:

```text
z = ... + b
```

Como el mismo bias afecta a todos los ejemplos, su gradiente acumula la senal de error de todos ellos:

```python
db2 = np.sum(dz2, axis=0, keepdims=True)
db1 = np.sum(dz1, axis=0, keepdims=True)
```

No multiplicamos por entradas porque el bias no multiplica ninguna entrada.

Su derivada local es `1`.

## 24. Por que el gradiente tiene la misma shape que el parametro

Cada elemento del parametro necesita su propia derivada.

Si:

```text
W1.shape = (2, 2)
```

entonces:

```text
dW1.shape = (2, 2)
```

Porque cada peso tiene una pregunta distinta:

```text
que pasa con la loss si cambio W1[0,0]?
que pasa con la loss si cambio W1[0,1]?
que pasa con la loss si cambio W1[1,0]?
que pasa con la loss si cambio W1[1,1]?
```

Lo mismo ocurre con:

```text
W2 -> dW2
b1 -> db1
b2 -> db2
```

## 25. Por que restamos el gradiente

El gradiente apunta hacia donde la loss sube mas rapido.

Si queremos bajar la loss, vamos en direccion contraria:

```python
W1 = W1 - LR * dW1
b1 = b1 - LR * db1
W2 = W2 - LR * dW2
b2 = b2 - LR * db2
```

`LR` controla el tamano del paso.

Si `LR` es demasiado pequeno, aprende muy lento.

Si `LR` es demasiado grande, puede saltarse el minimo, oscilar o divergir.

## 26. Que significa que un paso baje la perdida

El experimento registra:

```text
Loss antes de un paso:   0.2922916243810897
Loss despues de un paso: 0.28637467082417883
```

Esto significa que, con los gradientes calculados:

```text
W1 - LR*dW1
b1 - LR*db1
W2 - LR*dW2
b2 - LR*db2
```

produjo parametros ligeramente mejores para esa ejecucion inicial.

Es una prueba de sanidad importante.

## 27. Que significa la convergencia final

Predicciones finales:

```text
[[0.03908334],
 [0.95662095],
 [0.95598973],
 [0.03531293]]
```

Con umbral `0.5`:

```text
0.039 -> 0
0.956 -> 1
0.955 -> 1
0.035 -> 0
```

La red aprendio XOR:

```text
[[0],
 [1],
 [1],
 [0]]
```

La loss final:

```text
0.0016485673777804845
```

es pequena porque las predicciones estan cerca de las etiquetas.

## 28. Idea principal que hay que recordar

Backpropagation no es magia. Es la regla de la cadena aplicada de forma organizada.

El forward construye dependencias:

```text
X -> z1 -> a1 -> z2 -> y_pred -> L
```

El backward recorre esas dependencias al reves:

```text
L -> y_pred -> z2 -> a1 -> z1 -> parametros
```

Cada linea del backward responde una pregunta local:

```text
cuanto cambia esta variable si cambia la anterior?
```

Y al multiplicar esas respuestas locales, obtenemos:

```text
cuanto cambia la loss si cambio cada parametro?
```

Eso es exactamente lo que necesitamos para aprender.
