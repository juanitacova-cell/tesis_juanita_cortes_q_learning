# Q-learning en JMARKOV: caso de abastecimiento de fertilizantes

Código, datos y resultados del caso de estudio del proyecto de grado
**"Desarrollo de un módulo de Q-learning para la librería JMARKOV"**,
de Juanita Valentina Cortés Vargas (Ingeniería Industrial, Universidad de los Andes).
Asesor: Juan Fernando Pérez Bernal.

El módulo de Q-learning (`jmarkov/mdp/qlearning.py`), sus pruebas y los
ejercicios del curso Modelos de Decisión en el Tiempo están en el repositorio
oficial de JMARKOV: <https://github.com/copa-uniandes/jmarkov>
(rama `qlearning`, pull request #56). Este repositorio contiene solo el caso
de fertilizantes, que usa ese módulo.

## Estructura

```
data/
  raw/                         Excel con las series mensuales de precios
  processed/                   precios limpios y discretizados
src/
  01_limpieza_datos.py         limpieza de las series de precios
  02_analisis_descriptivo.py   estadísticas descriptivas y gráficas
  03_discretizacion.py         discretización de precios en categorías
  04_estimacion_transiciones.py  matrices de transición de precios
  05A_sdp_conjunto_jmarkov_EJECUTABLE.py      SDP conjunto de 5 épocas
  05B_sdp_independientes_jmarkov_REPO (1).py  SDP por insumo de 5 épocas
  06_mdp_value_iteration_independientes_REPO (1).py  MDP por insumo (Value Iteration)
  06A_mdp_conjunto_jmarkov_EJECUTABLE.py      MDP conjunto (no terminó en el tiempo disponible)
jmarkov/examples/qlearning_fertilizantes/
  qlearning_N.py, qlearning_P.py, qlearning_K.py   Q-learning por insumo frente al MDP
  sensibilidad_N.py, sensibilidad_P.py, sensibilidad_K.py   exploración inicial
  sensibilidad_ampliada.py     rejilla de alpha y epsilon por insumo
  qlearning_conjunto.py        Q-learning del modelo conjunto N-P-K
  sensibilidad_conjunto.py     rejilla del modelo conjunto
  qlearning_conjunto_capacidad.py  modelo conjunto con capacidad compartida
  graficas_sensibilidad.py     figuras de los resultados
outputs/
  tables/                      resultados en CSV
  figures/                     figuras en PNG
docs/                          notas de formulación
archivado/                     versión anterior del SDP, no usada en los resultados
```

La carpeta `jmarkov/examples/qlearning_fertilizantes/` conserva la ruta que
tenía dentro del repositorio de JMARKOV, para que los scripts encuentren
`src/` y `outputs/`. No contiene la librería: JMARKOV se instala aparte.

## Instalación

Requiere Python 3.8 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate             # en Mac o Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` instala JMARKOV desde GitHub en el commit que contiene el
módulo de Q-learning.

## Cómo reproducir los resultados

Desde la carpeta principal del repositorio:

```bash
python src/01_limpieza_datos.py
python src/02_analisis_descriptivo.py
python src/03_discretizacion.py
python src/04_estimacion_transiciones.py
python "src/05B_sdp_independientes_jmarkov_REPO (1).py"
python src/05A_sdp_conjunto_jmarkov_EJECUTABLE.py
python "src/06_mdp_value_iteration_independientes_REPO (1).py"

cd jmarkov
python examples/qlearning_fertilizantes/qlearning_N.py
python examples/qlearning_fertilizantes/qlearning_P.py
python examples/qlearning_fertilizantes/qlearning_K.py
python examples/qlearning_fertilizantes/sensibilidad_ampliada.py
python examples/qlearning_fertilizantes/qlearning_conjunto.py
python examples/qlearning_fertilizantes/sensibilidad_conjunto.py
python examples/qlearning_fertilizantes/qlearning_conjunto_capacidad.py
python examples/qlearning_fertilizantes/graficas_sensibilidad.py
```

Las rejillas (`sensibilidad_ampliada.py` y `sensibilidad_conjunto.py`) tardan
varias horas; sus resultados ya están en `outputs/tables/`.

## Relación con el documento

| Resultado | Script | Salida |
|---|---|---|
| Matrices de precios | `src/04_estimacion_transiciones.py` | `outputs/tables/transiciones_matriz_*_5_categorias.csv` |
| SDP por insumo y conjunto | `src/05B_...`, `src/05A_...` | `outputs/tables/sdp_jmarkov_*` |
| MDP por insumo (referencia) | `src/06_mdp_value_iteration_...` | `outputs/tables/mdp_value_iteration_*` |
| MDP conjunto (no terminó) | `src/06A_mdp_conjunto_...` | — |
| Q-learning por insumo | `qlearning_N.py`, `qlearning_P.py`, `qlearning_K.py` | `outputs/tables/qlearning_independientes/` |
| Exploración inicial | `sensibilidad_N.py`, `sensibilidad_P.py`, `sensibilidad_K.py` | `outputs/tables/qlearning_independientes/` |
| Rejilla por insumo y estabilidad | `sensibilidad_ampliada.py`, `graficas_sensibilidad.py` | `outputs/tables/qlearning_independientes/`, `outputs/figures/qlearning/` |
| Modelo conjunto | `qlearning_conjunto.py`, `sensibilidad_conjunto.py` | `outputs/tables/qlearning_conjunto/` |
| Capacidad compartida | `qlearning_conjunto_capacidad.py` | `outputs/tables/qlearning_conjunto_capacidad/` |