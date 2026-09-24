# Analisis del experimento

## Hipotesis

Responde: ¿que esperabas que ocurriera al entrenar un clasificador lineal sobre dos grupos sinteticos con `NOISE = 1.5`?

Los datos sinteticos tal y como estan planteados no suponen un gran reto para una red neuronal pequeña lo suficientemente entrenada, ya que de entrada muchos datos, aunque permutados, seguirán dentro de sus regiones correspondientes, por lo que de base el modelo no cuenta con un dataset demasiado disperso y complejo. Para ver un mal desempeño de la red neuronal hay que hacer serios cambios dentro de los parametros del experimento, como añadir mas ruido, etiquetas, solapamiento, etcétera. Ya que de entrada con los datos actuales del experimento (LR = 0.5, SAMPLES = 700, NOISE = 1.5) un clasificador lineal puede separar todos los datos de forma sencilla.


## Configuracion evaluada

Responde: ¿por que son relevantes `LR`, `EPOCHS`, `BATCH_SIZE`, el split train/validation y los umbrales `0.3`, `0.5` y `0.7`?

El LR es el tamaño del paso que utilizará el optimizador para ajustar pesos y bias en el backpropagation. Las Epochs son las pasadas que se harán sobre el conjunto entero de los datos, que en este caso esta distribuido en batches de tamaño 32, los batches son unicamente por hacer una actualizacion mas frecuente en el optimizador y por memoria, un batch pequeño tiene un gradiente mas ruidoso, pero en global ofrecen una estimacion aproximada de todo el gradiente del dataset. En otro experimento los epochs cuentan con un papel mas relevante ya que entra en juego el overfitting, pero al ser un clasificador lineal donde los datos no tienen ni demasiadas caracteristicas, y ademas tenemos muchos samples, en la práctica es imposible que el modelo sobreajuste,  además el ruido que hemos incluido en el dataset no es lo suficientemente grande como para que le modelo sobreajuste para encontrar regiones de separacion perfectas en todos los datos.

## Datos y separabilidad

Responde: ¿que muestra el scatter y el mapa de decision sobre el solapamiento entre las clases? ¿Por que una frontera lineal es una aproximacion razonable o limitada para estos datos?

El scatter nos muestra el ruido y un solapamiento entre clases tan grande como queramos indicar. Es interesante experimentar subiendo el Noise y fluctuando sobre el Learning Rate, ya que ahi podemos ver la influencia que tiene en las regiones de separacion del modelo, si tenemos un LR alto y un noise alto, el efecto de batches ruidosos puede generar que se inviertan las regiones de separacion, ya que generan un efecto en cadena, donde si los siguientes batches no corrigen el gradiente y estan tambien infectados o simplemente oscilan lentamente, las regiones y fronteras se veran afectadas. Por eso para el modelo es especialmente beneficioso siempre compensar el ruido con un LR bajo para que los batches ruidosos no incidan demasiado en la frontera. De esto inferimos que el clasificador lineal es sensible a datos anomalos y a sus parametros.


## Entrenamiento y validacion

Responde: ¿como evolucionan train loss y validation loss? ¿La diferencia final sugiere sobreajuste importante, una brecha moderada o estabilidad?

El train loss tiene una evolucion estable, esto era esperable dada la simpleza del clasificador y de la naturaleza de los datos. Lo mismo con el validation loss, con unos datos de tan baja dimensionalidad en la practica ver señales de overfitting es complicado dada la capcidad limitada del modelo, por lo cual el modelo siempre oscilará en los mismos parametros.

## Accuracy estandar

Responde: ¿que mide la accuracy con umbral `0.5`? ¿Que errores siguen presentes incluso si la accuracy parece razonable?

El umbral 0.5 es el umbral natural que le podemos dar al modelo para que separe los datos, esto es equivalente a decir que el coste de error de ambas clases es igual, ademas por el balanceo de las clases al modelo le resultará mas sencillo separar los datos. Lo mismo pasa con la accuracy, por estadística un clasificador constante que siempre predijera una clase tendria un accuracy del 50%, por lo que  toda precision superior a eso en mayor o menos medida nos indica que el modelo ha extraido una señal, el rendimiento ya vendrá dado por el mayor o menor indice de solapamiento

## Confianza y cobertura

Responde: compara `confidence_accuracy` con `coverage`. ¿Que se gana al aceptar solo predicciones con `p <= 0.3` o `p >= 0.7`? ¿Que se pierde?

confidence_accuracy no es mas que los datos correctamente clasificados en las regiones de confianza establecidas, condicionadas a todos los datos que satisfacen alguna de estas regiones, esten bien clasificados o no. El coverage son todos los datos que satisfacen alguna de las dos regiones, condicionados a todos los ejemplos de validacion actuales del batch. Al aceptar y medir estas regiones de confianza podemos evaluar si el modelo es seguro en sus predicciones, es decir, esto no implica que esté en lo correcto. Lo que perdemos son datos correctamente clasificados pero que no están dentro de la region optimista que aceptamos para clasificar clase 0 y clase 1, en términos de produccion los umbrales son mas o menos realizables dependiendo del dominio donde se vaya a desenvolver el modelo, si una mala prediccion resulta catastrófica los umbrales deben ser altos, sin embargo, si priorizamos volumen de predicciones (recall), lo ideal es mantener umbrales de confianza mas pesimistas (50%-60%). Básicamente el umbral alto puede dar muchos falsos negativos(en este caso abstenciones), pero con el beneficio de evaluar la seguridad del modelo.

## Limitaciones

Responde: identifica limites del experimento. Considera datos sinteticos, una sola semilla, un unico split, ausencia de test independiente y un modelo lineal.

Los limites del test están claros, la potencia del modelo y la naturaleza del dataset, es un modelo con una unica neurona donde los datos tienen baja dimensionalidad, la complejidad es muy reducida, ya que no puedes ver sobreajuste a los datos ni realmente tener que evaluar las características de estos.


## Siguiente experimento

Responde: ¿que pregunta deja abierta para `05_clasificacion_no_lineal_pytorch.py`?

En el siguiente experimento podemos apreciar un modelo de mayor complejidad ya que nos veremos a enfrentar datos de mayor dimensionalidad o complejidad, que nos harán diseñar un modelo mas profundo que genere fronteras mas complejas y precisas.