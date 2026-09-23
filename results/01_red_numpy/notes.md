# 01_red_numpy

## Pregunta

Puede una red minima implementada solo con NumPy aprender una relacion lineal del tipo `y = wx + b`?

## Hipotesis

Si el forward pass, la funcion de perdida, los gradientes y la actualizacion estan bien calculados, la perdida deberia bajar y los parametros aprendidos deberian acercarse a TRUE_W y TRUE_B.

## Configuracion

- Samples: 100
- TRUE_W: 3.0
- TRUE_B: 2.0
- Noise: 0.5
- Learning rate: 0.05
- Epochs: 100
- Seed: 42

## Resultados

- Loss inicial: 16.524267
- Loss final: 0.145558
- Mejor loss: 0.145558
- Epoch de mejor loss: 99
- w aprendido: 2.947557
- b aprendido: 1.974832
- Error absoluto en w: 0.052443
- Error absoluto en b: 0.025168

## Observaciones

- La perdida bajo durante el entrenamiento.
- Las perdidas son finitas.
- El modelo aprende una recta que debe compararse visualmente con la nube de puntos en `learned_line.png`.
- Como hay ruido en los datos, no se espera que la perdida llegue exactamente a cero.

## Conclusion

Este experimento demuestra el ciclo minimo de entrenamiento: forward pass, calculo de perdida, gradientes manuales y actualizacion de parametros mediante descenso de gradiente.

## Siguiente experimento

Probar una red pequena con capa oculta y backpropagation manual.
