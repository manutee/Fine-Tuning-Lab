# Fine-Tuning Lab

Fine-Tuning Lab es un proyecto educativo para aprender, paso a paso, como se entrenan, ajustan y evaluan modelos de machine learning y modelos de lenguaje.

La idea central es combinar matematicas, implementacion y experimentacion para comprender como aprenden los modelos, como se adaptan y como se evaluan las mejoras. La ruta conserva fine-tuning completo, LoRA, QLoRA e instruction tuning, y los conecta con aprendizaje estadistico, optimizacion, transformers, preentrenamiento y practica de investigacion.

El proyecto debe seguir siendo independiente, facil de leer y centrado en el aprendizaje. El tamano del modelo es una decision experimental; el progreso se mide por la capacidad de explicar, implementar, evaluar y cuestionar lo aprendido.

## Objetivo

Este repositorio sirve para estudiar conceptos, derivar sus operaciones esenciales, probarlos en notebooks, convertir lo aprendido en experimentos reproducibles y guardar resultados con conclusiones y limitaciones.

El objetivo a largo plazo es desarrollar una base solida en machine learning y aprendizaje profundo, con especializacion progresiva en modelos de lenguaje. La preparacion para investigar incluye leer articulos, reproducir resultados acotados y formular preguntas propias. Completar una lista de tecnicas no acredita por si solo dominio del campo.

La progresion esperada es:

```text
estudiar el concepto y su matematica
        ↓
formular una hipotesis y predecir el resultado
        ↓
implementar y explorar en un notebook
        ↓
convertirlo en un experimento reproducible
        ↓
comparar con una referencia y analizar errores
        ↓
guardar resultados, conclusiones y limitaciones
        ↓
extraer unicamente el codigo realmente reutilizable
```

## Filosofia educativa

- Cada experimento debe responder a una pregunta concreta.
- Cada script debe poder leerse de arriba abajo.
- Los hiperparametros deben estar visibles en el propio experimento.
- Las funciones son preferibles a las clases cuando sean suficientes.
- No se extrae codigo a `src/` hasta que exista reutilizacion real.
- Los notebooks se usan para estudiar y explorar.
- Los scripts de `experiments/` se usan para experimentos reproducibles.
- Los resultados deben incluir metricas y conclusiones, no solo archivos generados.
- Los tests se anaden cuando existan piezas importantes que merezcan proteccion.
- La estructura debe crecer solo cuando el codigo lo necesite.
- Las matematicas se estudian junto al problema que ayudan a entender, con ejercicios y derivaciones propias.
- Antes de ejecutar una variacion, se escribe que se espera observar y por que.
- Una mejora debe compararse con una referencia sencilla y bajo condiciones documentadas.
- Los resultados negativos y las discrepancias con la hipotesis tambien forman parte del aprendizaje.
- Se avanza cuando se puede explicar, reconstruir y modificar lo esencial con autonomia.

## Que no pretende ser

Este proyecto no es una plataforma de experimentacion ni una arquitectura empresarial.

No incluye, de momento:

- CLI.
- API.
- Dashboard.
- Base de datos.
- Microservicios.
- Sistema de plugins.
- Factories, repositories o inyeccion de dependencias.
- Configuraciones YAML complejas.
- Orquestadores de experimentos.
- Tracking externo.
- Pipelines completos de entrenamiento.
- Integraciones con modelos grandes o servicios externos.

## Estructura

```text
fine-tuning-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── README.md
├── experiments/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── training.py
│   └── evaluation.py
├── results/
│   └── README.md
└── tests/
    └── README.md
```

## Notebooks, Experiments y Src

`notebooks/` contiene exploracion: ideas incompletas, derivaciones, visualizaciones, pruebas pequenas y aprendizaje conceptual.

`experiments/` contiene scripts reproducibles: una pregunta, una configuracion visible, una ejecucion clara, metricas guardadas y una conclusion.

`src/` contiene solo codigo reutilizable. No debe convertirse en un lugar para esconder la logica principal de aprendizaje. Si una funcion solo se usa en un experimento, puede vivir primero en ese experimento.

## Hoja de ruta

### Estado actual y orden de recorrido

Estamos en la **etapa 2, cerca de pasar a la etapa 3**. La etapa 1 es la base ya trabajada; sus conceptos se repasaran cuando haga falta. Las etapas posteriores describen trabajo previsto, no implementaciones terminadas.

Se conservan las etapas originales 1-6 y su numeracion. Se incorporan dos bloques complementarios y se desarrollan las etapas posteriores:

| Orden | Bloque | Proposito |
| --- | --- | --- |
| 1 | Etapas 1 y 2 | Fundamentos y entrenamiento explicito |
| 2 | Bloque A, junto al cierre de la etapa 2 y durante la 3 | Aprendizaje estadistico y optimizacion |
| 3 | Etapa 3 | Primera experiencia de fine-tuning de un encoder |
| 4 | Bloque B, despues de la etapa 3 y antes de profundizar en LoRA | Transformer pequeno y preentrenamiento |
| 5 | Etapas 4, 5 y 6 | LoRA y QLoRA con comparaciones de calidad y coste |
| 6 | Etapas 7 y 8 | Instruction tuning, preferencias y razonamiento con RL |
| 7 | Etapas 9 y 10 | Sistemas, especializacion y reproduccion de investigacion |

La etapa 3 sigue siendo el siguiente proyecto principal. Para empezarla se necesita comprender el ciclo de entrenamiento, la separacion de datos y la evaluacion; las demas piezas del bloque A se profundizan durante el proyecto. La matematica y el metodo experimental se trabajan de forma continua.

### Etapa 1 - Fundamentos con NumPy

Conceptos:

- neurona;
- capa lineal;
- funciones de activacion;
- funcion de perdida;
- forward pass;
- derivadas;
- backpropagation;
- descenso de gradiente;
- comparacion con PyTorch.

Trabajo principalmente en notebooks: regresion lineal, red pequena y XOR.

Matematica asociada: operaciones matriciales, derivadas parciales, regla de la cadena y significado del gradiente.

Criterio de comprension: reconstruir el forward y el backward, justificar las dimensiones y comprobar gradientes con diferencias finitas y con autograd. Aprender los cuatro puntos de XOR verifica el ajuste de esos puntos; no demuestra generalizacion.

### Etapa 2 - Training loop con PyTorch

Objetivos:

- dataset sencillo;
- dataloader;
- modelo pequeno;
- forward;
- loss;
- backward;
- optimizer step;
- entrenamiento y validacion;
- curvas de perdida;
- seeds y reproducibilidad.

El training loop se escribe progresivamente durante el aprendizaje, manteniendo visibles sus operaciones.

Matematica asociada: interpretacion de logits, sigmoid, MSE y entropia cruzada binaria; derivacion de la perdida binaria desde una distribucion Bernoulli; gradientes por minibatch y estabilidad numerica.

Criterio para pasar a la etapa 3: poder entrenar y validar un modelo pequeno, explicar `zero_grad`, `backward` y `step`, distinguir `train`/`eval` de la desactivacion de gradientes, y detectar sobreajuste mediante curvas. Se debe poder explicar por que los datos de evaluacion no participan en el ajuste de pesos.

### Bloque A - Aprendizaje estadistico y optimizacion

Se inicia al cerrar la etapa 2 y acompana la etapa 3.

Conceptos y ejercicios:

- probabilidad condicional, esperanza, varianza y maxima verosimilitud;
- separacion entre entrenamiento, validacion y test; fuga de informacion;
- generalizacion, sobreajuste, sesgo, varianza y regularizacion;
- metricas ante clases desbalanceadas, calibracion e incertidumbre de la evaluacion;
- SGD, momentum y Adam: reglas de actualizacion, learning rate y efecto del batch size;
- inicializacion, saturacion, normas de gradientes y condicionamiento;
- referencias sencillas: regresion lineal/logistica y comparacion con un arbol o ensemble cuando el problema lo justifique;
- proyecciones y PCA como introduccion a representaciones y reduccion de dimensionalidad.

Experimentos: variar cantidad de datos, ruido, capacidad y regularizacion; comparar optimizadores con un presupuesto documentado. Repetir comparaciones relevantes con varias semillas y estudiar su variabilidad.

Criterio de comprension: distinguir un problema de optimizacion de uno de generalizacion, justificar una metrica y explicar que evidencia falta antes de afirmar que un modelo es mejor.

### Etapa 3 - Fine-tuning de un encoder pequeno

Comparaciones futuras:

- entrenamiento solo de la cabeza;
- ultimas N capas;
- fine-tuning completo.

Metricas y observaciones futuras:

- accuracy;
- macro F1;
- tiempo;
- parametros entrenables;
- convergencia;
- uso de memoria cuando sea posible.

El dataset se elegira cuando llegue esta etapa. No se descarga ninguno ahora.

Conceptos asociados: transferencia de aprendizaje, representaciones, tokenizacion, embeddings, mascaras de atencion y funcionamiento general de un encoder. Se estudiara que informacion recibe cada posicion y como se obtiene una prediccion para la tarea.

Comparacion experimental: mantener las particiones y el protocolo de evaluacion, incluir una referencia sencilla y documentar el presupuesto de ajuste de cada estrategia. Analizar errores por clase y posibles efectos del desbalance.

Criterio de comprension: explicar que parametros cambian, que aporta el preentrenamiento y que evidencia permite elegir entre entrenar la cabeza, las ultimas capas o todo el modelo.

### Bloque B - Transformer pequeno y preentrenamiento

Se desarrolla despues de la primera experiencia con un encoder y antes de profundizar en las etapas 4-6.

Objetivos:

- construir un modelo de lenguaje pequeno con PyTorch y operaciones visibles;
- estudiar tokenizacion y construir o analizar un tokenizador sencillo;
- implementar embeddings, atencion escalada, multiples cabezas y mascara causal;
- comprender posiciones, conexiones residuales, normalizacion y capas feed-forward;
- formular la prediccion del siguiente token, preparar entradas y etiquetas desplazadas y calcular entropia cruzada;
- seleccionar, limpiar y separar datos; detectar duplicados entre particiones;
- preentrenar a pequena escala y evaluar perdida, perplexity y muestras generadas;
- estudiar muestreo, temperatura y diferencias entre entrenamiento e inferencia.

Matematica asociada: productos escalares, softmax, maxima verosimilitud, entropia y divergencia KL. Relacionar las operaciones con sus gradientes y su coste de memoria y computo.

Experimentos: retirar o modificar un componente cada vez, comparar cantidades de datos o tamanos de modelo y observar como cambian perdida, coste y errores. Una ablacion es una comparacion que retira o modifica un componente para estudiar su contribucion.

Criterio de comprension: seguir el recorrido desde los tokens hasta la perdida, explicar la mascara causal y justificar una comparacion bajo un presupuesto limitado. Los resultados de un modelo pequeno no se extrapolan automaticamente a modelos grandes.

### Etapa 4 - LoRA educativo

Objetivos futuros:

- comprender `W + BA`;
- implementar una capa `LoRALinear`;
- congelar los pesos originales;
- estudiar rango, alpha y dropout;
- merge y unmerge;
- comparar con PEFT.

Primero se implementara manualmente en un caso pequeno.

Matematica asociada: rango, factorizacion matricial, valores singulares y restriccion `Delta W = BA`. Derivar los gradientes de los factores y contar parametros entrenables.

Criterio de comprension: justificar que restringe el rango, comprobar la equivalencia del merge en condiciones deterministas y comparar LoRA con fine-tuning completo bajo un protocolo documentado.

### Etapa 5 - LoRA en un decoder de 0.5-1.5B

Objetivos futuros:

- usar un modelo generativo;
- aplicar LoRA donde tenga utilidad real;
- comparar modulos objetivo;
- comparar rangos;
- medir memoria y parametros entrenables;
- evaluar generacion.

El rango de 0.5-1.5B es orientativo y depende del hardware disponible. Se estudiaran la perdida sobre los tokens objetivo, las mascaras de etiquetas y la diferencia entre adaptar un modelo y preentrenarlo.

Criterio de comprension: justificar los modulos y rangos elegidos, analizar ejemplos de mejora y degradacion y medir calidad junto a memoria y tiempo.

### Etapa 6 - QLoRA en un modelo de alrededor de 3B

Objetivos futuros:

- cuantizacion a 4 bits;
- QLoRA;
- gradient checkpointing;
- packing;
- comparacion de memoria, tiempo y calidad.

El tamano de alrededor de 3B es orientativo. Se estudiaran el error de cuantizacion, los tipos numericos, los estados del optimizador y las activaciones para explicar el consumo de memoria. El packing debe mantener explicitas las decisiones sobre fronteras entre ejemplos y mascaras.

Criterio de comprension: explicar que se cuantiza, que se entrena y que memoria sigue siendo necesaria; comparar con una referencia de mayor precision cuando los recursos lo permitan y documentar las limitaciones en caso contrario.

### Etapa 7 - Instruction tuning y aprendizaje de preferencias

Objetivos:

- instruction tuning supervisado: datos, formato de conversacion y tokens sobre los que se calcula la perdida;
- calidad y diversidad de instrucciones, separacion de evaluacion y analisis de errores;
- preference tuning: pares de respuestas, supuestos y calidad de las preferencias;
- modelos de recompensa, papel de una politica de referencia y regularizacion mediante KL;
- DPO: derivar e implementar su objetivo en un caso pequeno;
- comparar modelo base, ajuste supervisado y ajuste por preferencias.

Criterio de comprension: explicar que senal aprende cada objetivo y por que mejorar una metrica de preferencias no garantiza mejorar todas las capacidades. Evaluar tambien degradaciones y sesgos del evaluador.

### Etapa 8 - Aprendizaje por refuerzo y razonamiento

Objetivos:

- fundamentos de RL: politica, trayectoria, recompensa, retorno y ventaja;
- derivar un estimador de gradiente de politica y estudiar su varianza;
- comprender el papel de baselines y restricciones a la actualizacion;
- estudiar PPO y variantes de optimizacion con grupos de respuestas segun la pregunta elegida;
- entrenar un caso pequeno con recompensas verificables y compararlo con ajuste supervisado;
- analizar explotacion de errores de la recompensa y diferencias entre exito medido y tarea real;
- comparar estrategias de generacion y verificacion con presupuesto de inferencia controlado.

Criterio de comprension: distinguir mas aprendizaje de mas computo al responder, justificar la recompensa y detectar cuando el sistema mejora el indicador sin resolver mejor el problema.

### Etapa 9 - Inferencia optimizada y entrenamiento distribuido

Objetivos:

- profiling: medir donde se gasta el tiempo y la memoria;
- mixed precision, acumulacion de gradientes y checkpointing;
- batching, KV cache y coste de generar tokens;
- estudiar implementaciones eficientes de atencion y sus compromisos;
- entrenamiento distribuido: paralelismo de datos, sincronizacion de gradientes y reparto de estados;
- comparar rendimiento, memoria y equivalencia numerica cuando corresponda.

Las primeras mediciones de eficiencia empiezan en etapas anteriores. Esta etapa profundiza en sistemas; las ejecuciones con varias GPU dependen de los recursos disponibles y se distinguen de los ejercicios conceptuales.

Criterio de comprension: identificar un cuello de botella con medidas y explicar por que una optimizacion mejora o empeora el resultado bajo esas condiciones.

### Etapa 10 - Especializacion y reproduccion de investigacion

Elegir una direccion: datos y preentrenamiento, optimizacion, razonamiento con RL, interpretabilidad o eficiencia de sistemas. La eleccion determina que matematicas y herramientas se profundizan.

Proceso:

1. Leer un articulo y separar hipotesis, metodo, evidencia y limitaciones.
2. Elegir un resultado concreto que pueda reproducirse con los recursos disponibles.
3. Implementar una referencia y documentar diferencias respecto al trabajo original.
4. Reproducir la comparacion con un protocolo definido y analizar variabilidad.
5. Realizar una ablacion o plantear una extension pequena.
6. Escribir un informe que explique resultados, discrepancias y preguntas abiertas.

Se practica la lectura de articulos desde etapas anteriores. Aqui la unidad principal de trabajo pasa a ser una pregunta de investigacion acotada.

Criterio de comprension: defender una conclusion con evidencia y explicar que experimento podria refutarla. Una reproduccion parcial debe identificarse como tal.

## Matematicas a lo largo de la ruta

Cada bloque combina intuicion, derivacion y comprobacion numerica. Se reservan sesiones para resolver problemas sin consultar la solucion y se retoman las bases cuando aparece una dificultad.

| Area | Contenidos | Conexion principal |
| --- | --- | --- |
| Algebra lineal | Rango, bases, proyecciones, normas, autovalores y SVD | Representaciones, condicionamiento y LoRA |
| Calculo multivariable | Gradientes, jacobianos, regla de la cadena y hessianos | Backpropagation y curvatura de la perdida |
| Probabilidad | Distribuciones, condicionamiento, esperanza y verosimilitud | Perdidas y modelado de datos |
| Estadistica | Estimacion, sesgo, varianza e incertidumbre | Generalizacion y comparacion de resultados |
| Optimizacion y calculo numerico | Metodos de gradiente, estabilidad y precision finita | Entrenamiento y eficiencia |
| Teoria de la informacion | Entropia, entropia cruzada y KL | Modelos de lenguaje y preferencias |

No es necesario dominar toda la tabla para pasar a la etapa 3. Cada concepto se introduce donde resulta util y se profundiza mediante ejercicios y experimentos.

## Forma de trabajo

Cuando se desarrolle un archivo de entrenamiento, el flujo recomendado sera:

1. Formular la pregunta y escribir una prediccion razonada.
2. Explicar el objetivo matematico, las operaciones y las dimensiones relevantes.
3. Dividir el problema en pasos pequenos; proponer firmas simples si ayudan.
4. Intentar la derivacion y la implementacion antes de consultar una solucion.
5. Comprobar operaciones esenciales con un caso pequeno o una referencia fiable.
6. Ejecutar la comparacion y analizar curvas, errores y discrepancias.
7. Revisar el codigo y corregir decisiones a partir de la evidencia.
8. Escribir conclusiones propias y reconstruir o modificar lo esencial sin mirar.

La ayuda de IA se utiliza para pedir explicaciones, pistas, revision y preguntas de comprobacion. Las soluciones completas se piden cuando son necesarias; despues se debe poder reconstruir la pieza central y explicar sus decisiones. Leer codigo generado y reconocer su logica no basta como criterio de dominio.

## Protocolo experimental

La exigencia se adapta al ejercicio. Una comprobacion de una derivada puede necesitar un caso determinista; una afirmacion de mejora entre modelos requiere una comparacion mas amplia.

- Definir antes de entrenar la pregunta, la metrica principal y la referencia de comparacion.
- Separar entrenamiento, validacion y test cuando se estudie generalizacion; ajustar decisiones con validacion y reservar el test para la evaluacion final.
- Evitar fuga de informacion en preprocesamiento, seleccion de datos y duplicados entre particiones.
- Documentar datos, configuracion, semillas, versiones y hardware relevantes.
- Cambiar una variable principal en las ablaciones. Si se cambian varias, limitar la atribucion del resultado.
- Explicitar el presupuesto: pasos, tokens, tiempo o computo, segun la pregunta. Registrar el coste del ajuste de hiperparametros.
- Repetir comparaciones importantes con varias semillas cuando sea viable y reportar dispersion. Una sola ejecucion se presenta como evidencia preliminar.
- Acompanar las metricas de ejemplos, errores, limitaciones y explicaciones alternativas.
- Guardar tambien los resultados que contradicen la hipotesis.

## Entorno

Crear un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las dependencias iniciales no estan fijadas a versiones concretas para evitar restricciones prematuras. Cuando un experimento necesite reproducibilidad estricta, se podran fijar versiones documentadas.

Dependencias como `transformers`, `datasets`, `peft`, `accelerate`, `trl`, `bitsandbytes`, `mlflow`, `wandb`, `fastapi`, `hydra` o `deepspeed` se anadiran solo cuando una etapa las necesite.

## Ejecutar notebooks

```powershell
jupyter notebook
```

Los notebooks deben vivir en `notebooks/`. Cuando un notebook se estabilice y responda a una pregunta experimental concreta, conviene convertirlo en un script dentro de `experiments/`.

## Ejecutar tests

```powershell
pytest
```

Al inicio puede no haber tests ejecutables. Se anadiran cuando existan piezas importantes: metricas, operaciones matematicas, checkpointing, reproducibilidad o equivalencias como el merge de LoRA.

## Anadir un experimento

Un nuevo experimento deberia:

1. Definir una pregunta o hipotesis, una prediccion y una referencia de comparacion.
2. Mostrar la configuracion y el presupuesto al principio del archivo.
3. Cargar los datos de forma explicita.
4. Cargar o definir el modelo.
5. Preparar la estrategia de entrenamiento.
6. Entrenar.
7. Evaluar con el protocolo definido y analizar errores y variabilidad cuando corresponda.
8. Guardar resultados en `results/<nombre_del_experimento>/`.
9. Escribir en `notes.md` conclusiones, limitaciones, discrepancias y el siguiente experimento que ayudaria a resolverlas.

El codigo comun solo debe moverse a `src/` cuando ya haya repeticion real.

## Regla de implementacion

Este repositorio no debe incluir entrenamientos completos generados automaticamente antes de haberlos estudiado.

El objetivo es escribir y comprender de forma progresiva las operaciones matematicas, la carga de datos, tokenizacion, dataloaders, forward pass, perdida, backward pass, optimizador, validacion, metricas, arquitectura del transformer, preentrenamiento, checkpointing, acumulacion de gradientes, mixed precision, LoRA educativo, objetivos de preferencias y benchmarks.

