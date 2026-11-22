# Estrategia computacional con jMarkov: Modelo A y Modelo B

## Modelo A — SDP conjunto N-P-K

La formulación principal se mantiene conjunta. En cada época `t` el estado es

\[
S_t=(I_{N,t},I_{P,t},I_{K,t},p_{N,t},p_{P,t},p_{K,t}),
\]

y la acción es

\[
a_t=(a_{N,t},a_{P,t},a_{K,t}).
\]

La dinámica de inventario es determinística:

\[
I_{c,t+1}=I_{c,t}+a_{c,t}-d_c,
\]

con satisfacción obligatoria de demanda y capacidad máxima. Bajo independencia condicional de los precios,

\[
P(j\mid i,a)=P_N(p'_N\mid p_N)P_P(p'_P\mid p_P)P_K(p'_K\mid p_K)
\]

si los inventarios siguientes coinciden con los determinados por la acción; en otro caso, la probabilidad es cero.

El costo base es

\[
c_t(i,a)=\bar p_N(p_N)a_N+\bar p_P(p_P)a_P+\bar p_K(p_K)a_K,
\]

con costos de almacenamiento iguales a cero. El término de ventas queda pendiente de definición.

### Representación que exigiría `jmarkov.sdp.dtsdp`

- `periods`: arreglo de las 5 épocas.
- `states`: arreglo con todos los estados conjuntos.
- `actions`: arreglo global con todas las acciones conjuntas.
- `transition_matrices`: diccionario de matrices densas por época y acción.
- `immediate_returns`: arreglo de dimensión `épocas × estados × acciones`.
- `discount_factor`: factor de descuento.

Con una malla de 5% de la demanda, los tres inventarios tienen 3 niveles. Al excluir la categoría 5 de fósforo por no tener transición saliente y ser inalcanzable, el modelo tiene

\[
3^3\times 10\times 9\times 10=24,300
\]

estados. Cada commodity tiene 5 compras globales posibles en la unión de acciones factibles, por lo que el espacio global contiene hasta

\[
5^3=125
\]

acciones.

Una sola matriz densa de transición tendría `24,300 × 24,300` entradas, aproximadamente 4.40 GiB en `float64`. Para 125 acciones y cuatro transiciones entre cinco épocas, la representación densa superaría aproximadamente 2.1 TiB. Por ello el script `05A_sdp_conjunto_jmarkov_NO_EJECUTAR.py` construye la metadata y cuantifica memoria, pero deliberadamente no materializa dichas matrices ni llama a `solve()`.

Este cuello de botella es relevante para el objetivo posterior de modificar/extender jMarkov para espacios de estado grandes, por ejemplo mediante matrices dispersas o generación de estados alcanzables.

## Modelo B — tres instancias independientes en jMarkov

Como benchmark y validación se resuelven tres SDP:

\[
S_t^N=(I_{N,t},p_{N,t}),\qquad
S_t^P=(I_{P,t},p_{P,t}),\qquad
S_t^K=(I_{K,t},p_{K,t}).
\]

Para cada commodity,

\[
V_t^c(I_c,p_c)=\min_{a_c\in A_c(I_c)}\left\{
\bar p_c(p_c)a_c+\beta\sum_{p'_c}P_c(p'_c\mid p_c)
V_{t+1}^c(I'_c,p'_c)
\right\}.
\]

El script `05B_sdp_independientes_jmarkov.py` usa `jmarkov.sdp.dtsdp` y mantiene el arreglo global de acciones requerido por la librería. Las acciones que no son factibles en un estado reciben una penalización muy grande y una transición self-loop para conservar matrices estocásticas.

### Ajuste numérico por la implementación inspeccionada

La versión de `dtsdp.solve()` inspeccionada transforma internamente los retornos cuando `minimize=True` y utiliza valores iniciales finitos. Para evitar problemas con costos de decenas de millones de USD:

1. se define retorno como el negativo del costo;
2. se divide por `1,000,000` para trabajar en millones de USD;
3. se usa `solve(minimize=False)`.

Estas transformaciones no cambian la política óptima.

La documentación de jMarkov exige un factor de descuento menor que uno. El benchmark usa `beta=0.999999`, muy cercano a uno, para aproximar el caso no descontado de cinco meses. Este punto debe validarse con el asesor.

## Relación entre A y B

El Modelo B no reemplaza el Modelo A. Bajo los supuestos actuales de costos aditivos, capacidades ya repartidas e independencia de precios, los tres subproblemas son separables y sirven como benchmark. Si posteriormente se introduce una restricción conjunta —por ejemplo presupuesto compartido o capacidad común— la separabilidad desaparece y el modelo conjunto se vuelve indispensable.

### Nota adicional sobre etiquetas de acciones

En el código de `dtsdp.solve()` inspeccionado, la matriz de decisiones se crea con `dtype=str` sin longitud explícita. En NumPy esto puede truncar cadenas largas a un solo carácter. Por esa razón el benchmark usa etiquetas de acción de una sola letra (`A`, `B`, `C`, `D`, `E`) y mantiene un diccionario interno que las traduce a toneladas compradas.
