# Desglose matematico de `02_backpropagation_numpy`

Este documento explica con detalle la matematica del experimento XOR implementado en:

```text
notebooks/02_backpropagation_numpy.py
```

El objetivo no es memorizar formulas, sino entender que significa cada variable, por que cada gradiente tiene esa forma y como se relacionan las variables durante forward y backward.

La red que estás construyendo es:

$$
\boxed{2 \rightarrow 2 \rightarrow 1}
$$

Es decir:

* 2 entradas: \(x_1,x_2\)
* 2 neuronas ocultas
* 1 neurona de salida

y entrenamos los cuatro ejemplos XOR simultáneamente.

---

# 1. Antes de derivar nada: la arquitectura matemática

Tus datos son:

$$
X =
\begin{bmatrix}
0&0\\
1&0\\
0&1\\
1&1
\end{bmatrix}
$$

y

$$
y =
\begin{bmatrix}
0\\
1\\
1\\
0
\end{bmatrix}
$$

Por tanto:

$$
X\in\mathbb{R}^{4\times2}
$$

porque tenemos:

* 4 ejemplos
* 2 características por ejemplo.

Y:

$$
y\in\mathbb{R}^{4\times1}
$$

La red hace esencialmente esto:

$$
X
\rightarrow Z_1
\rightarrow A_1
\rightarrow Z_2
\rightarrow \hat y
\rightarrow L
$$

con:

$$
Z_1=XW_1+b_1
$$

$$
A_1=\sigma(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
\hat y=\sigma(Z_2)
$$

y finalmente:

$$
L=\frac1n\sum_{i=1}^n(\hat y_i-y_i)^2
$$

---

# 2. Las shapes de toda la red

Este es probablemente el punto más importante antes de hablar de derivadas.

Tus matrices tienen:

$$
X:(4,2)
$$

$$
W_1:(2,2)
$$

$$
b_1:(1,2)
$$

$$
W_2:(2,1)
$$

$$
b_2:(1,1)
$$

Podemos dibujarlo así:

```text
X             W1              A1             W2           y_pred

(4,2)   @    (2,2)    ->    (4,2)    @    (2,1)   ->    (4,1)

4 ejemplos      ↑                2 neuronas       ↑          1 salida
2 features      │                ocultas          │
                │                                  │
            parámetros                         parámetros
```

La regla de multiplicación matricial es:

$$
(m\times n)(n\times p)=(m\times p)
$$

Las dimensiones interiores tienen que coincidir.

Por eso:

$$
(4\times2)(2\times2)
=
(4\times2)
$$

y después:

$$
(4\times2)(2\times1)
=
(4\times1)
$$

Esto no es casualidad. **Las shapes de los pesos están determinadas por el número de neuronas de cada capa.**

---

# 3. Qué representa realmente \(W_1\)

Tienes:

```python
W1 = rng.normal(..., size=(2, 2))
```

Con tu semilla concreta obtenemos aproximadamente:

$$
W_1=
\begin{bmatrix}
0.15236 & -0.51999\\
0.37523 & 0.47028
\end{bmatrix}
$$

Hay que entender muy bien qué significa cada elemento.

Podemos etiquetarlo:

$$
W_1=
\begin{bmatrix}
w_{11}&w_{12}\\
w_{21}&w_{22}
\end{bmatrix}
$$

La **columna 1** contiene los pesos que llegan a la neurona oculta 1.

La **columna 2** contiene los pesos que llegan a la neurona oculta 2.

Así:

```text
x1 ── w11 ──> h1
x2 ── w21 ──> h1

x1 ── w12 ──> h2
x2 ── w22 ──> h2
```

Esto explica por qué \(W_1\) tiene shape:

$$
(\text{nº entradas},\text{nº neuronas})
$$

es decir:

$$
(2,2)
$$

---

# 4. Forward: primera capa

Tu código:

```python
z1 = X @ W1 + b1
```

Matemáticamente:

$$
Z_1=XW_1+b_1
$$

Empecemos pensando en **un solo ejemplo**:

$$
x=[x_1,x_2]
$$

entonces:

$$
[x_1,x_2]
\begin{bmatrix}
w_{11}&w_{12}\\
w_{21}&w_{22}
\end{bmatrix}
$$

produce:

$$
[
x_1w_{11}+x_2w_{21},
x_1w_{12}+x_2w_{22}
]
$$

Luego añadimos los bias:

$$
z_1^{(1)}
=
x_1w_{11}+x_2w_{21}+b_{11}
$$

$$
z_1^{(2)}
=
x_1w_{12}+x_2w_{22}+b_{12}
$$

Son exactamente las combinaciones lineales de las dos neuronas ocultas.

---

# 5. Hagámoslo con uno de tus ejemplos

Tomemos:

$$
x=[1,0]
$$

Con:

$$
W_1=
\begin{bmatrix}
0.15236&-0.51999\\
0.37523&0.47028
\end{bmatrix}
$$

y inicialmente:

$$
b_1=[0,0]
$$

tenemos:

$$
z_1
=
[1,0]
\begin{bmatrix}
0.15236&-0.51999\\
0.37523&0.47028
\end{bmatrix}
$$

Primera neurona:

$$
1(0.15236)+0(0.37523)=0.15236
$$

Segunda:

$$
1(-0.51999)+0(0.47028)=-0.51999
$$

Por tanto:

$$
z_1=[0.15236,-0.51999]
$$

---

# 6. ¿Qué hace sigmoid?

Después haces:

```python
a1 = sigmoid(z1)
```

donde:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Aplicándolo elemento a elemento:

$$
A_1=\sigma(Z_1)
$$

Para nuestro ejemplo:

$$
\sigma(0.15236)\approx0.53802
$$

$$
\sigma(-0.51999)\approx0.37285
$$

Entonces:

$$
a_1=[0.53802,0.37285]
$$

Esta es la salida de las dos neuronas ocultas.

---

# 7. Lo interesante: haces los 4 ejemplos simultáneamente

NumPy no calcula cada ejemplo por separado.

Hace:

$$
XW_1
$$

completo.

Con tus valores iniciales:

$$
Z_1=
\begin{bmatrix}
0&0\\
0.15236&-0.51999\\
0.37523&0.47028\\
0.52758&-0.04971
\end{bmatrix}
$$

Shape:

$$
(4,2)
$$

Cada fila corresponde a un ejemplo.

Cada columna a una neurona oculta.

Después:

$$
A_1=\sigma(Z_1)
$$

obteniendo aproximadamente:

$$
A_1=
\begin{bmatrix}
0.50000&0.50000\\
0.53802&0.37285\\
0.59272&0.61545\\
0.62892&0.48758
\end{bmatrix}
$$

Otra vez:

$$
A_1:(4,2)
$$

Aquí ya aparece una idea central de deep learning:

> una matriz representa simultáneamente un batch completo de ejemplos.

---

# 8. Segunda capa

Ahora haces:

```python
z2 = a1 @ W2 + b2
```

Tienes:

$$
A_1:(4,2)
$$

y:

$$
W_2:(2,1)
$$

Con tu seed:

$$
W_2=
\begin{bmatrix}
-0.97552\\
-0.65109
\end{bmatrix}
$$

Por tanto:

$$
Z_2=A_1W_2+b_2
$$

Shape:

$$
(4,2)(2,1)=(4,1)
$$

Tiene muchísimo sentido.

Cada ejemplo tiene dos activaciones ocultas:

$$
[a_{1},a_{2}]
$$

y queremos combinarlas para producir **una única salida**.

Para el ejemplo anterior:

$$
a_1=[0.53802,0.37285]
$$

hacemos:

$$
z_2
=
0.53802(-0.97552)
+
0.37285(-0.65109)
$$

aproximadamente:

$$
z_2=-0.76761
$$

Después:

$$
\hat y=\sigma(z_2)
$$

$$
\hat y\approx0.3170
$$

---

# 9. Las predicciones iniciales completas

Tu red empieza produciendo aproximadamente:

$$
\hat y=
\begin{bmatrix}
0.30719\\
0.31700\\
0.27311\\
0.28273
\end{bmatrix}
$$

pero debería producir:

$$
y=
\begin{bmatrix}
0\\
1\\
1\\
0
\end{bmatrix}
$$

Por eso necesitamos entrenarla.

---

# 10. La función de pérdida MSE

Tu código:

```python
error = y_pred - y
return np.mean(error**2)
```

Matemáticamente:

$$
L
=
\frac1n
\sum_{i=1}^{n}
(\hat y_i-y_i)^2
$$

Aquí:

$$
n=4
$$

Por tanto:

$$
L=
\frac14
[
(\hat y_1-y_1)^2+
(\hat y_2-y_2)^2+
(\hat y_3-y_3)^2+
(\hat y_4-y_4)^2
]
$$

Con las predicciones iniciales:

$$
L\approx0.29229
$$

---

# 11. Ahora empieza lo realmente importante: backpropagation

Queremos responder a esta pregunta:

> ¿Cómo cambia el loss si modifico ligeramente cada peso?

Es decir, queremos:

$$
\frac{\partial L}{\partial W_1},
\quad
\frac{\partial L}{\partial b_1},
\quad
\frac{\partial L}{\partial W_2},
\quad
\frac{\partial L}{\partial b_2}
$$

Eso son tus:

```python
dW1
db1
dW2
db2
```

---

# 12. El grafo matemático completo

Tu forward es:

$$
W_1
\rightarrow
Z_1
\rightarrow
A_1
\rightarrow
Z_2
\rightarrow
\hat Y
\rightarrow
L
$$

En sentido forward:

```text
X
 \
  -> Z1 -> A1 -> Z2 -> y_pred -> Loss
 /             /
W1            W2
```

Backpropagation recorre exactamente el camino contrario:

```text
Loss
 ↓
y_pred
 ↓
z2
 ↓
W2 / a1
      ↓
      z1
      ↓
      W1
```

Por eso tu código hace:

```python
dloss_dypred
dz2
dW2
da1
dz1
dW1
```

No es una colección arbitraria de fórmulas.

Es literalmente **recorrer al revés el grafo del forward usando la regla de cadena**.

---

# 13. Regla de cadena: el concepto fundamental

Supón:

$$
L=f(y)
$$

pero:

$$
y=g(z)
$$

y:

$$
z=h(w)
$$

Entonces:

$$
L=f(g(h(w)))
$$

Queremos:

$$
\frac{dL}{dw}
$$

La regla de cadena dice:

$$
\boxed{
\frac{dL}{dw}
=
\frac{dL}{dy}
\frac{dy}{dz}
\frac{dz}{dw}
}
$$

Esta fórmula es esencialmente **todo backpropagation**.

Para tu segunda capa:

$$
L
\rightarrow
\hat y
\rightarrow
z_2
\rightarrow
W_2
$$

Así:

$$
\boxed{
\frac{\partial L}{\partial W_2}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z_2}
\frac{\partial z_2}{\partial W_2}
}
$$

Eso es exactamente lo que implementa tu código.

---

# 14. Primera derivada: MSE respecto a la predicción

Tenemos:

$$
L=
\frac1n
\sum_i(\hat y_i-y_i)^2
$$

Para un ejemplo:

$$
L_i=\frac1n(\hat y_i-y_i)^2
$$

Derivamos respecto a \(\hat y_i\):

$$
\frac{\partial L_i}{\partial \hat y_i}
=
\frac1n
\frac{\partial}{\partial\hat y_i}
(\hat y_i-y_i)^2
$$

Aplicamos:

$$
\frac{d}{dx}x^2=2x
$$

Entonces:

$$
\boxed{
\frac{\partial L}{\partial\hat y_i}
=
\frac{2}{n}(\hat y_i-y_i)
}
$$

Que es exactamente:

```python
error = y_pred - y
dloss_dypred = 2.0 * error / n
```

Shape:

$$
(4,1)
$$

---

# 15. Tus valores concretos

El error es:

$$
\hat y-y
=
\begin{bmatrix}
0.30719\\
-0.68300\\
-0.72689\\
0.28273
\end{bmatrix}
$$

Como:

$$
\frac2n=\frac24=0.5
$$

obtenemos:

$$
\frac{\partial L}{\partial\hat y}
=
\begin{bmatrix}
0.15359\\
-0.34150\\
-0.36345\\
0.14136
\end{bmatrix}
$$

Observa algo interesante.

Para el primer ejemplo:

$$
y=0,\quad\hat y=0.307
$$

el gradiente es positivo.

Eso está diciendo:

> para reducir el loss, sería conveniente hacer descender esta predicción.

Para el segundo:

$$
y=1,\quad\hat y=0.317
$$

el gradiente es negativo.

Eso indica que deberíamos aumentar la predicción.

---

# 16. Derivada de sigmoid

Tu función es:

$$
\sigma(z)
=
\frac1{1+e^{-z}}
$$

Y escribiste:

```python
def Derivada_Activacion(a):
    return a * (1-a)
```

¿Por qué?

Vamos a derivarlo.

Partimos de:

$$
\sigma(z)=(1+e^{-z})^{-1}
$$

Derivamos:

$$
\sigma'(z)
=
-(1+e^{-z})^{-2}
\cdot
(-e^{-z})
$$

por lo que:

$$
\sigma'(z)
=
\frac{e^{-z}}
{(1+e^{-z})^2}
$$

Podemos reescribir:

$$
\frac1{1+e^{-z}}
\left(
\frac{e^{-z}}{1+e^{-z}}
\right)
$$

El primer término es:

$$
\sigma(z)
$$

y el segundo:

$$
1-\sigma(z)
$$

Por tanto:

$$
\boxed{
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
}
$$

Como ya guardaste:

$$
a=\sigma(z)
$$

no necesitas volver a calcular sigmoid.

Simplemente:

$$
\boxed{
\sigma'(z)=a(1-a)
}
$$

De ahí:

```python
return a * (1-a)
```

---

# 17. Regla de cadena entre loss y \(z_2\)

Tenemos:

$$
\hat y=\sigma(z_2)
$$

Sabemos:

$$
\frac{\partial L}{\partial\hat y}
$$

y:

$$
\frac{\partial\hat y}{\partial z_2}
=
\hat y(1-\hat y)
$$

Por regla de cadena:

$$
\boxed{
\frac{\partial L}{\partial z_2}
=
\frac{\partial L}{\partial\hat y}
\odot
\frac{\partial\hat y}{\partial z_2}
}
$$

Uso \(\odot\) porque aquí estamos haciendo una **multiplicación elemento a elemento**, no matricial.

Tu código:

```python
dypred_dz2 = Derivada_Activacion(y_pred)

dz2 = dloss_dypred * dypred_dz2
```

Es muy importante distinguir:

```python
*
```

de:

```python
@
```

En NumPy:

$$
*
$$

es producto elemento a elemento.

Mientras que:

$$
@
$$

es multiplicación matricial.

---

# 18. Shapes aquí

Tenemos:

$$
\frac{\partial L}{\partial\hat y}:(4,1)
$$

y:

$$
\frac{\partial\hat y}{\partial z_2}:(4,1)
$$

Entonces:

```python
(4,1) * (4,1)
```

produce:

$$
(4,1)
$$

Tus valores aproximadamente son:

$$
dz_2=
\begin{bmatrix}
0.03269\\
-0.07394\\
-0.07215\\
0.02867
\end{bmatrix}
$$

Aquí:

$$
dz_2 \equiv \frac{\partial L}{\partial Z_2}
$$

---

# 19. Ahora viene una de las partes más bonitas: ¿por qué `a1.T @ dz2`?

Tu código:

```python
dW2 = a1.T @ dz2
```

Puede parecer casi magia la primera vez.

Pero vamos a derivarlo.

Tenemos:

$$
Z_2=A_1W_2+b_2
$$

Pensemos primero en un ejemplo.

La salida antes de sigmoid es:

$$
z_2
=
a_{11}w_1+
a_{12}w_2+b
$$

Queremos:

$$
\frac{\partial L}{\partial w_1}
$$

Por regla de cadena:

$$
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial z_2}
\frac{\partial z_2}{\partial w_1}
$$

Pero:

$$
\frac{\partial z_2}{\partial w_1}
=
a_{11}
$$

Por tanto:

$$
\frac{\partial L}{\partial w_1}
=
a_{11}
\frac{\partial L}{\partial z_2}
$$

Análogamente:

$$
\frac{\partial L}{\partial w_2}
=
a_{12}
\frac{\partial L}{\partial z_2}
$$

---

# 20. Pero tenemos cuatro ejemplos

Cada ejemplo contribuye al mismo peso.

Así que:

$$
\frac{\partial L}{\partial w_1}
=
a_{11}^{(1)}\delta_1
+
a_{11}^{(2)}\delta_2
+
a_{11}^{(3)}\delta_3
+
a_{11}^{(4)}\delta_4
$$

donde:

$$
\delta_i=\frac{\partial L}{\partial z_{2,i}}
$$

Esto se puede escribir como un producto escalar:

$$
\begin{bmatrix}
a_{11}^{(1)}&
a_{11}^{(2)}&
a_{11}^{(3)}&
a_{11}^{(4)}
\end{bmatrix}
\begin{bmatrix}
\delta_1\\
\delta_2\\
\delta_3\\
\delta_4
\end{bmatrix}
$$

Y ahí aparece:

$$
A_1^T dz_2
$$

---

# 21. Mira las shapes

Originalmente:

$$
A_1:(4,2)
$$

Entonces:

$$
A_1^T:(2,4)
$$

Y:

$$
dz_2:(4,1)
$$

Por tanto:

$$
(2,4)(4,1)=(2,1)
$$

exactamente la shape de:

$$
W_2:(2,1)
$$

Por eso:

```python
dW2 = a1.T @ dz2
```

produce:

$$
dW_2:(2,1)
$$

Una propiedad muy útil es:

$$
\boxed{
\operatorname{shape}
\left(
\frac{\partial L}{\partial W}
\right)
=
\operatorname{shape}(W)
}
$$

Siempre.

---

# 22. El resultado numérico inicial

En tu primera iteración:

$$
dW_2
\approx
\begin{bmatrix}
-0.04817\\
-0.04165
\end{bmatrix}
$$

Esto significa:

$$
\frac{\partial L}{\partial W_{2,1}}
\approx -0.04817
$$

$$
\frac{\partial L}{\partial W_{2,2}}
\approx -0.04165
$$

---

# 23. ¿Y por qué el bias usa una suma?

Tienes:

```python
db2 = np.sum(dz2, axis=0, keepdims=True)
```

Recordemos:

$$
z_2=a_1W_2+b_2
$$

Para cada ejemplo:

$$
\frac{\partial z_2}{\partial b_2}=1
$$

Por tanto:

$$
\frac{\partial L}{\partial b_2}
=
\sum_i
\frac{\partial L}{\partial z_{2,i}}
$$

Es decir:

$$
\boxed{
db_2=\sum_i dz_{2,i}
}
$$

Aquí:

$$
db_2\approx -0.08473
$$

Shape:

$$
(1,1)
$$

igual que:

$$
b_2:(1,1)
$$

---

# 24. Ahora tenemos que llevar el error hacia atrás

Hasta ahora hemos llegado aquí:

```text
Loss
  ↓
y_pred
  ↓
 z2
 ↓
W2
```

Pero para modificar \(W_1\) necesitamos continuar:

```text
z2
 ↓
a1
 ↓
z1
 ↓
W1
```

Tenemos:

$$
z_2=A_1W_2+b_2
$$

y queremos:

$$
\frac{\partial L}{\partial A_1}
$$

Tu código:

```python
da1 = dz2 @ W2.T
```

Veamos por qué.

---

# 25. Derivación de `dz2 @ W2.T`

Para un solo ejemplo:

$$
z_2=a_1^{(1)}w_1+a_1^{(2)}w_2+b
$$

Por tanto:

$$
\frac{\partial z_2}{\partial a_1^{(1)}}=w_1
$$

y:

$$
\frac{\partial z_2}{\partial a_1^{(2)}}=w_2
$$

Aplicando regla de cadena:

$$
\frac{\partial L}{\partial a_1^{(1)}}
=
\frac{\partial L}{\partial z_2}
w_1
$$

$$
\frac{\partial L}{\partial a_1^{(2)}}
=
\frac{\partial L}{\partial z_2}
w_2
$$

En vector:

$$
\frac{\partial L}{\partial A_1}
=
dz_2W_2^T
$$

---

# 26. Shapes de esta propagación

Tenemos:

$$
dz_2:(4,1)
$$

$$
W_2:(2,1)
$$

por tanto:

$$
W_2^T:(1,2)
$$

y:

$$
(4,1)(1,2)
=
(4,2)
$$

exactamente la shape de:

$$
A_1:(4,2)
$$

Así:

```python
da1 = dz2 @ W2.T
```

produce:

$$
da_1:(4,2)
$$

Este paso es muy importante conceptualmente:

> el error de una única neurona de salida se reparte hacia las dos neuronas ocultas de acuerdo con los pesos que las conectaban con ella.

Si un peso \(W_2\) es grande, esa neurona oculta tenía mucha influencia en la salida y recibe proporcionalmente más gradiente.

---

# 27. Tus valores concretos

Inicialmente:

$$
W_2^T=
[-0.97552,-0.65109]
$$

y:

$$
dz_2=
\begin{bmatrix}
0.03269\\
-0.07394\\
-0.07215\\
0.02867
\end{bmatrix}
$$

Entonces:

$$
da_1\approx
\begin{bmatrix}
-0.03189&-0.02128\\
0.07213&0.04814\\
0.07038&0.04698\\
-0.02797&-0.01867
\end{bmatrix}
$$

Shape:

$$
(4,2)
$$

---

# 28. Ahora volvemos a atravesar una sigmoid

Recordemos:

$$
A_1=\sigma(Z_1)
$$

Tenemos:

$$
\frac{\partial L}{\partial A_1}
$$

pero necesitamos:

$$
\frac{\partial L}{\partial Z_1}
$$

Por regla de cadena:

$$
\frac{\partial L}{\partial Z_1}
=
\frac{\partial L}{\partial A_1}
\odot
\frac{\partial A_1}{\partial Z_1}
$$

Y:

$$
\frac{\partial A_1}{\partial Z_1}
=
A_1(1-A_1)
$$

Entonces:

$$
\boxed{
dZ_1
=
dA_1\odot A_1(1-A_1)
}
$$

Tu código:

```python
da1_dz1 = Derivada_Activacion(a1)
dz1 = da1 * da1_dz1
```

Exactamente eso.

Las shapes:

$$
da_1:(4,2)
$$

$$
a_1(1-a_1):(4,2)
$$

por lo que:

$$
dz_1:(4,2)
$$

---

# 29. Valores iniciales

Obtienes aproximadamente:

$$
dz_1=
\begin{bmatrix}
-0.00797&-0.00532\\
0.01793&0.01126\\
0.01699&0.01112\\
-0.00653&-0.00466
\end{bmatrix}
$$

Esto contiene:

$$
\frac{\partial L}{\partial z_{1,ij}}
$$

para cada ejemplo \(i\) y cada neurona oculta \(j\).

---

# 30. Finalmente llegamos a \(W_1\)

Tenemos:

$$
Z_1=XW_1+b_1
$$

Queremos:

$$
\frac{\partial L}{\partial W_1}
$$

Exactamente igual que hicimos con \(W_2\):

$$
\boxed{
dW_1=X^Tdz_1
}
$$

Tu código:

```python
dW1 = X.T @ dz1
```

Shapes:

$$
X:(4,2)
$$

Entonces:

$$
X^T:(2,4)
$$

y:

$$
dz_1:(4,2)
$$

Por tanto:

$$
(2,4)(4,2)=(2,2)
$$

que coincide exactamente con:

$$
W_1:(2,2)
$$

---

# 31. ¿Qué está sumando realmente esa multiplicación?

Esta es una parte que merece detenerse.

Tenemos:

$$
X^T=
\begin{bmatrix}
0&1&0&1\\
0&0&1&1
\end{bmatrix}
$$

y:

$$
dZ_1=
\begin{bmatrix}
\delta_{11}&\delta_{12}\\
\delta_{21}&\delta_{22}\\
\delta_{31}&\delta_{32}\\
\delta_{41}&\delta_{42}
\end{bmatrix}
$$

Entonces:

$$
dW_1=X^TdZ_1
$$

produce:

$$
\begin{bmatrix}
0&1&0&1\\
0&0&1&1
\end{bmatrix}
\begin{bmatrix}
\delta_{11}&\delta_{12}\\
\delta_{21}&\delta_{22}\\
\delta_{31}&\delta_{32}\\
\delta_{41}&\delta_{42}
\end{bmatrix}
$$

Resultado:

$$
\begin{bmatrix}
\delta_{21}+\delta_{41}
&
\delta_{22}+\delta_{42}
\\
\delta_{31}+\delta_{41}
&
\delta_{32}+\delta_{42}
\end{bmatrix}
$$

¿Por qué?

Porque solamente los ejemplos donde una entrada vale 1 contribuyen al gradiente del peso correspondiente.

---

# 32. Tu \(dW_1\) inicial

Obtienes:

$$
dW_1
\approx
\begin{bmatrix}
0.01140&0.00659\\
0.01046&0.00645
\end{bmatrix}
$$

Y:

$$
db_1
\approx
\begin{bmatrix}
0.02042&0.01239
\end{bmatrix}
$$

Fíjate otra vez:

$$
dW_1:(2,2)
$$

igual que:

$$
W_1:(2,2)
$$

y:

$$
db_1:(1,2)
$$

igual que:

$$
b_1:(1,2)
$$

---

# 33. Podemos condensar todo el backward en cuatro ecuaciones

Toda tu función `backward()` esencialmente calcula esto:

## Salida

$$
\boxed{
dZ_2
=
\frac{2}{n}(\hat Y-Y)
\odot
\hat Y(1-\hat Y)
}
$$

Después:

$$
\boxed{
dW_2=A_1^TdZ_2
}
$$

$$
\boxed{
db_2=\sum_{\text{batch}}dZ_2
}
$$

## Capa oculta

$$
\boxed{
dA_1=dZ_2W_2^T
}
$$

$$
\boxed{
dZ_1
=
dA_1
\odot
A_1(1-A_1)
}
$$

Finalmente:

$$
\boxed{
dW_1=X^TdZ_1
}
$$

$$
\boxed{
db_1=\sum_{\text{batch}}dZ_1
}
$$

Eso es todo tu backpropagation.

---

# 34. El mapa de shapes completo

Este esquema te recomiendo tenerlo muy presente:

```text
FORWARD
=======

X
(4,2)

  @ W1
    (2,2)
  -------
Z1
(4,2)

 sigmoid
  ↓

A1
(4,2)

  @ W2
    (2,1)
  -------
Z2
(4,1)

 sigmoid
  ↓

Y_pred
(4,1)

  ↓

Loss
scalar
```

Y ahora al revés:

```text
BACKWARD
========

Loss
  ↓

dY_pred
(4,1)

  * sigmoid'(Z2)
  ↓

dZ2
(4,1)

        ┌──────────────┐
        │              │
        ↓              ↓

A1.T @ dZ2       dZ2 @ W2.T

(2,4)(4,1)       (4,1)(1,2)

     ↓                 ↓

dW2                  dA1
(2,1)                (4,2)

                       *
                 sigmoid'(Z1)
                       ↓

                     dZ1
                     (4,2)

                       ↓

                 X.T @ dZ1

                (2,4)(4,2)

                       ↓

                     dW1
                     (2,2)
```

---

# 35. Una regla extraordinariamente útil para recordar las fórmulas

Para una capa genérica:

$$
Z=AW+b
$$

si durante backpropagation conoces:

$$
dZ=\frac{\partial L}{\partial Z}
$$

entonces siempre tienes:

$$
\boxed{dW=A^TdZ}
$$

$$
\boxed{db=\sum dZ}
$$

$$
\boxed{dA=dZW^T}
$$

Estas tres ecuaciones aparecen continuamente en redes neuronales.

En tu segunda capa:

$$
Z_2=A_1W_2+b_2
$$

entonces:

$$
dW_2=A_1^TdZ_2
$$

$$
db_2=\sum dZ_2
$$

$$
dA_1=dZ_2W_2^T
$$

En tu primera capa:

$$
Z_1=XW_1+b_1
$$

entonces:

$$
dW_1=X^TdZ_1
$$

$$
db_1=\sum dZ_1
$$

Y podrías incluso calcular:

$$
dX=dZ_1W_1^T
$$

aunque no lo necesitas porque \(X\) no es un parámetro entrenable.

---

# 36. ¿De dónde salen las transpuestas?

Esta es una duda fundamental.

No están puestas simplemente para "hacer cuadrar shapes".

Aparecen naturalmente al derivar la multiplicación matricial.

Si:

$$
Z=AW
$$

las tres relaciones diferenciales fundamentales son:

$$
\boxed{
dW=A^TdZ
}
$$

y:

$$
\boxed{
dA=dZW^T
}
$$

Observa la simetría:

```text
Forward:

A @ W
  ↓
  Z


Backward hacia W:

A.T @ dZ
   ↓
   dW


Backward hacia A:

dZ @ W.T
   ↓
   dA
```

Esto merece memorizarse, pero **después de entender de dónde viene**, no como una fórmula arbitraria.

---

# 37. La intuición del producto matricial durante backpropagation

Puedes pensar que `dZ` contiene:

> cuánto importa cada neurona para el error.

Entonces:

```python
A.T @ dZ
```

pregunta:

> ¿cuánto contribuyó cada entrada de esa neurona a ese error?

Mientras:

```python
dZ @ W.T
```

pregunta:

> ¿cuánto del error debo devolver a cada neurona anterior, teniendo en cuenta cuánto influía sobre esta neurona?

Es una forma bastante buena de desarrollar intuición.

---

# 38. Actualización mediante gradient descent

Una vez calculados los gradientes haces:

```python
W1 = W1 - LR * dW1
```

etc.

Matemáticamente:

$$
W_1^{nuevo}
=
W_1^{viejo}
-
\eta
\frac{\partial L}{\partial W_1}
$$

donde:

$$
\eta=LR=0.5
$$

¿Por qué restamos?

Porque el gradiente apunta en la dirección de **máximo crecimiento** de la función.

Si:

$$
\nabla L
$$

apunta hacia donde \(L\) aumenta más rápido, entonces:

$$
-\nabla L
$$

apunta hacia donde disminuye.

Por eso:

$$
\boxed{
\theta\leftarrow\theta-\eta\nabla_\theta L
}
$$

---

# 39. Ejemplo con uno de tus pesos

Inicialmente:

$$
W_{1,11}=0.15235854
$$

y encontramos:

$$
\frac{\partial L}{\partial W_{1,11}}
\approx0.01140124
$$

Con:

$$
LR=0.5
$$

el cambio es:

$$
0.5(0.01140124)=0.00570062
$$

Entonces:

$$
W_{1,11}^{nuevo}
=
0.15235854-0.00570062
$$

$$
\boxed{
W_{1,11}^{nuevo}\approx0.14665792
}
$$

La red acaba de modificar ligeramente ese peso en una dirección que localmente debería reducir el error.

Y haces esto con **todos los parámetros simultáneamente**.

---

# 40. Una epoch completa matemáticamente

Cada iteración de:

```python
for epoch in range(Epochs):
```

realiza exactamente:

### 1. Forward

$$
Z_1=XW_1+b_1
$$

$$
A_1=\sigma(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
\hat Y=\sigma(Z_2)
$$

### 2. Loss

$$
L=
\frac1n
\|\hat Y-Y\|_2^2
$$

### 3. Backward

$$
dZ_2=
\frac2n(\hat Y-Y)
\odot
\hat Y(1-\hat Y)
$$

$$
dW_2=A_1^TdZ_2
$$

$$
db_2=\sum dZ_2
$$

$$
dA_1=dZ_2W_2^T
$$

$$
dZ_1=dA_1\odot A_1(1-A_1)
$$

$$
dW_1=X^TdZ_1
$$

$$
db_1=\sum dZ_1
$$

### 4. Gradient descent

$$
W_1\leftarrow W_1-\eta dW_1
$$

$$
b_1\leftarrow b_1-\eta db_1
$$

$$
W_2\leftarrow W_2-\eta dW_2
$$

$$
b_2\leftarrow b_2-\eta db_2
$$

Y vuelves a empezar.

---

# 41. Un detalle profundo: aquí el batch size es 4

Tú no estás entrenando ejemplo por ejemplo.

Estás pasando:

$$
X:(4,2)
$$

completo.

Por tanto estás haciendo **full-batch gradient descent**.

Los cuatro ejemplos participan en cada actualización de pesos.

Eso explica por qué aparecen sumas como:

$$
A_1^TdZ_2
$$

La multiplicación matricial está acumulando simultáneamente la contribución al gradiente de los cuatro ejemplos.

Con millones de ejemplos normalmente se emplean minibatches:

$$
X_{\text{batch}}
\in
\mathbb R^{B\times D}
$$

por ejemplo:

$$
B=32,\quad D=784
$$

Pero la matemática sería exactamente la misma.

---

# 42. El patrón general escala a redes enormes

Lo interesante es que lo que estás haciendo aquí con:

```text
2 → 2 → 1
```

es conceptualmente lo mismo que ocurre en redes con millones de parámetros.

Supón una capa:

$$
X:(64,768)
$$

$$
W:(768,3072)
$$

Entonces:

$$
Z=XW
$$

produce:

$$
(64,768)(768,3072)
=
(64,3072)
$$

Durante backward, si:

$$
dZ:(64,3072)
$$

entonces:

$$
dW=X^TdZ
$$

$$
(768,64)(64,3072)
=
(768,3072)
$$

Exactamente la misma fórmula que:

```python
dW1 = X.T @ dz1
```

en tu pequeña red XOR.

Esto es precisamente lo valioso de hacer este experimento desde NumPy: estás viendo directamente la matemática que los frameworks esconden.

---

# 43. Diferencia entre `*` y `@` en tu red

En tu código aparecen dos operaciones completamente distintas.

## Producto elemento a elemento

```python
dz2 = dloss_dypred * dypred_dz2
```

Aquí:

$$
\begin{bmatrix}
a\\b\\c
\end{bmatrix}
\odot
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
=
\begin{bmatrix}
ax\\by\\cz
\end{bmatrix}
$$

No mezclamos neuronas ni ejemplos.

Cada elemento afecta a su correspondiente elemento.

Es la regla de cadena elemento a elemento.

---

## Producto matricial

```python
dW2 = a1.T @ dz2
```

Aquí sí tenemos:

$$
C_{ij}
=
\sum_k A_{ik}B_{kj}
$$

Es decir, cada elemento resultante contiene una **suma de productos**.

Esta operación sirve para agregar cómo distintos ejemplos/neuronas contribuyeron a un determinado peso.

---

# 44. Y el broadcasting de los biases

También ocurre algo interesante aquí:

```python
z1 = X @ W1 + b1
```

Tenemos:

$$
XW_1:(4,2)
$$

pero:

$$
b_1:(1,2)
$$

NumPy hace broadcasting.

Es conceptualmente como si:

$$
b_1=
[b_1,b_2]
$$

se replicase cuatro veces:

$$
\begin{bmatrix}
b_1&b_2\\
b_1&b_2\\
b_1&b_2\\
b_1&b_2
\end{bmatrix}
$$

y después:

$$
Z_1=XW_1+B
$$

Eso explica también por qué al hacer backward debemos sumar sobre las filas:

```python
db1 = np.sum(dz1, axis=0, keepdims=True)
```

Como un mismo bias se utilizó para los cuatro ejemplos, su gradiente recibe contribución de los cuatro.

---

# 45. `axis=0` tiene un significado matemático

Tienes:

$$
dz_1:(4,2)
$$

por ejemplo:

$$
\begin{bmatrix}
a&b\\
c&d\\
e&f\\
g&h
\end{bmatrix}
$$

Cuando haces:

```python
np.sum(dz1, axis=0)
```

eliminas la dimensión de las filas:

$$
[a+c+e+g,\;b+d+f+h]
$$

Por tanto:

$$
(4,2)\rightarrow(2,)
$$

Con:

```python
keepdims=True
```

mantienes:

$$
(1,2)
$$

que coincide con:

$$
b_1:(1,2)
$$

---

# 46. Una manera muy poderosa de razonar sobre las shapes

Cuando estés implementando redes manualmente, puedes comprobar constantemente:

$$
\boxed{
\text{shape}(d\theta)=\text{shape}(\theta)
}
$$

Así:

| Parámetro |   Shape | Gradiente |   Shape |
| --------- | ------: | --------- | ------: |
| \(W_1\)   | `(2,2)` | \(dW_1\)  | `(2,2)` |
| \(b_1\)   | `(1,2)` | \(db_1\)  | `(1,2)` |
| \(W_2\)   | `(2,1)` | \(dW_2\)  | `(2,1)` |
| \(b_2\)   | `(1,1)` | \(db_2\)  | `(1,1)` |

Si esto no ocurre, casi seguramente tienes un error en el backward.

---

# 47. Regla de cadena completa para un peso concreto

Vamos ahora a llevarlo al extremo, porque aquí es donde realmente se entiende backprop.

Supón que queremos saber:

$$
\frac{\partial L}{\partial w_{11}^{(1)}}
$$

es decir, un peso de la primera capa.

El camino hasta el loss es:

$$
w_{11}^{(1)}
\rightarrow
z_{1,1}
\rightarrow
a_{1,1}
\rightarrow
z_2
\rightarrow
\hat y
\rightarrow
L
$$

Así:

$$
\boxed{
\frac{\partial L}{\partial w_{11}^{(1)}}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z_2}
\frac{\partial z_2}{\partial a_{1,1}}
\frac{\partial a_{1,1}}{\partial z_{1,1}}
\frac{\partial z_{1,1}}{\partial w_{11}^{(1)}}
}
$$

Cada término tiene sentido.

Sabemos que:

$$
\frac{\partial L}{\partial\hat y}
=
\frac2n(\hat y-y)
$$

$$
\frac{\partial\hat y}{\partial z_2}
=
\hat y(1-\hat y)
$$

$$
\frac{\partial z_2}{\partial a_{1,1}}
=
w_{2,1}
$$

$$
\frac{\partial a_{1,1}}{\partial z_{1,1}}
=
a_{1,1}(1-a_{1,1})
$$

$$
\frac{\partial z_{1,1}}{\partial w_{11}^{(1)}}
=
x_1
$$

Entonces:

$$
\boxed{
\frac{\partial L}{\partial w_{11}^{(1)}}
=
\frac2n(\hat y-y)
\hat y(1-\hat y)
w_{2,1}
a_{1,1}(1-a_{1,1})
x_1
}
$$

Para un solo ejemplo.

Después sumamos las contribuciones de todos los ejemplos.

Y esto es exactamente lo que tus operaciones matriciales calculan de golpe.

---

# 48. Esto es probablemente la idea central de todo el código

Las matrices **no cambian la matemática de las derivadas**.

Simplemente permiten calcular miles o millones de expresiones como:

$$
\frac{\partial L}{\partial w}
$$

simultáneamente.

Es decir:

> backpropagation puede entenderse escalar por escalar, pero se implementa matricialmente porque hacerlo así es enormemente más eficiente.

Tu código es ambas cosas a la vez:

```python
dz2 = ...
dW2 = a1.T @ dz2
da1 = dz2 @ W2.T
dz1 = ...
dW1 = X.T @ dz1
```

Cada línea compacta potencialmente cientos, miles o millones de derivadas escalares.

---

# 49. La red completa en una única expresión

Incluso podríamos escribir tu red como:

$$
\boxed{
\hat Y
=
\sigma
\left(
\sigma(XW_1+b_1)W_2+b_2
\right)
}
$$

Y el loss:

$$
\boxed{
L(W_1,b_1,W_2,b_2)
=
\frac1n
\left\|
\sigma
\left(
\sigma(XW_1+b_1)W_2+b_2
\right)
-Y
\right\|^2
}
$$

Cuando llamas a:

```python
backward(...)
```

lo que realmente estás haciendo es calcular:

$$
\nabla L
=
\left[
\frac{\partial L}{\partial W_1},
\frac{\partial L}{\partial b_1},
\frac{\partial L}{\partial W_2},
\frac{\partial L}{\partial b_2}
\right]
$$

de esa gigantesca función compuesta.

---

# 50. La jerarquía mental que te recomiendo

Para entender redes neuronales matemáticamente, yo separaría lo que estás aprendiendo en estas **cinco capas conceptuales**:

1. **Escalar:** entender una sola neurona:

   $$
   z=w_1x_1+w_2x_2+b
   $$

2. **Regla de cadena:** entender:

   $$
   \frac{\partial L}{\partial w}
   =
   \frac{\partial L}{\partial a}
   \frac{\partial a}{\partial z}
   \frac{\partial z}{\partial w}
   $$

3. **Una capa completa:** entender:

   $$
   Z=XW+b
   $$

4. **Forma matricial del backward:**

   $$
   dW=X^TdZ,\qquad
   dX=dZW^T
   $$

5. **Una red completa:** encadenar todas las capas anteriores.

Tu código ya contiene las cinco. Esa es precisamente la razón por la que es un buen ejercicio.

---

## Tu `backward()` traducido literalmente a matemáticas

Finalmente, puedes leer tu función así:

```python
error = y_pred - y
dloss_dypred = 2.0 * error/n
```

$$
\frac{\partial L}{\partial\hat Y}
=
\frac2n(\hat Y-Y)
$$

```python
dypred_dz2 = y_pred * (1-y_pred)
dz2 = dloss_dypred * dypred_dz2
```

$$
dZ_2=
\frac{\partial L}{\partial\hat Y}
\odot
\sigma'(Z_2)
$$

```python
dW2 = a1.T @ dz2
```

$$
dW_2=A_1^TdZ_2
$$

```python
db2 = np.sum(dz2, axis=0)
```

$$
db_2=\sum_i dZ_{2,i}
$$

```python
da1 = dz2 @ W2.T
```

$$
dA_1=dZ_2W_2^T
$$

```python
dz1 = da1 * a1*(1-a1)
```

$$
dZ_1=dA_1\odot\sigma'(Z_1)
$$

```python
dW1 = X.T @ dz1
```

$$
dW_1=X^TdZ_1
$$

```python
db1 = np.sum(dz1, axis=0)
```

$$
db_1=\sum_i dZ_{1,i}
$$

Ese bloque de unas diez líneas es, esencialmente, **todo el algoritmo de backpropagation para tu red**.

En el siguiente nivel útil de este experimento, yo iría todavía más despacio y cogería **solo el ejemplo \(X=(1,0), y=1\)**, eliminando temporalmente el batch. Podemos calcular a mano, número por número, los dos valores de \(z_1\), las dos sigmoides, \(z_2\), \(\hat y\), el MSE y después **cada una de las derivadas hasta llegar a los cuatro pesos**, dibujando simultáneamente el grafo computacional. Después volveríamos a introducir las matrices `(4,2)` y verías exactamente cómo NumPy convierte esas cuatro cuentas individuales en `X.T @ dz1`. Esa sería la mejor forma de hacer que la regla de cadena y las multiplicaciones matriciales dejen de parecer fórmulas memorizadas y se vuelvan intuitivas.




