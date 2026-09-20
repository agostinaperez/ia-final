# Detección Automática de Eventos Epilépticos en Señales EEG mediante CNN-1D

## 1. Definición del Problema
La inspección visual de registros de electroencefalogramas (EEG) resulta un proceso complejo y demandante para los neurólogos debido a la magnitud de los datos en monitoreos continuos. El objetivo es implementar una arquitectura de Inteligencia Artificial (Red Neuronal Convolucional o CNN 1D) capaz de clasificar ventanas temporales de señales crudas de EEG para diferenciar automáticamente entre períodos de crisis (actividad ictal) frente a períodos de actividad normal (basal).

### 1.1. Formulación del agente según el framework PEAS
Para especificar formalmente al agente, se utiliza el framework PEAS, y se define:

- **Performance**: clasificar correctamente cada ventana temporal de EEG como "crisis" o "no crisis". Se cuantifica con métricas clínicas como Sensibilidad (Recall), Especificidad y Tasa de Falsas Alarmas (FPR). El objetivo es maximizar la sensibilidad manteniendo el FPR bajo.
- **Environment**: el contexto clínico de monitoreo EEG continuo. Está constituido por la actividad eléctrica cerebral del paciente, captada en canales a 256 Hz (la cantidad de canales se normaliza a 16 al momento de preprocesar, y los mismos se estandarizan en canales bipolares), junto con el ruido y los artefactos propios del registro. Es un entorno no estacionario, parcialmente observable (la señal cruda incluye ruido/artefactos que ocultan la señal útil) y con alto volumen de datos.
- **Actuators**: el agente no actúa físicamente sobre el paciente; es un sistema de apoyo a la decisión. Su acción es emitir la clasificación binaria (crisis / no crisis) y, en un despliegue real, generar una alarma o aviso al personal médico cuando se detecta una crisis.
- **Sensors**: los electrodos del montaje, que miden diferencias de potencial sobre el cuero cabelludo. La señal cruda de estos sensores, digitalizada a 256 Hz y filtrada en 0.5–50 Hz, es la entrada que percibe el modelo.

## 2. Origen y Naturaleza de los Datos
Se utiliza un dataset clínico de dominio público perteneciente al Hospital Infantil de Boston (CHB) en colaboración con el MIT, conformando el CHB-MIT Scalp EEG Database(https://physionet.org/content/chbmit/1.0.0/).
Estas son señales temporales continuas, altamente no estacionarias y afectadas por múltiples tipos de artefactos.
A nivel de preprocesamiento, es necesario un filtrado digital pasabanda (0.5 a 50 Hz) para aislar la frecuencia neuronal útil, y una división en ventanas temporales fijas de 5.12 segundos (con 50% de solapamiento) para asegurar la continuidad temporal de las ventanas contiguas.

El entorno utiliza Python >= 3.14.4 Se fija una semilla aleatoria (seed) global para asegurar el determinismo en la división de lotes y entrenamiento.

## 3. Métricas de Éxito y Baseline
- Baseline Actual: Inspección visual manual (humana) que resulta en cargas analíticas excesivas y alta vulnerabilidad a la fatiga en monitoreos prolongados.
- Métricas de Éxito Clínico: Más allá de la exactitud global (Accuracy), la eficacia del sistema se medirá mediante:
- Sensibilidad (Recall): Capacidad de detectar correctamente las ventanas con crisis.
- Especificidad: Capacidad de excluir actividad de fondo o ruido.
- Tasa de Falsas Alarmas (FPR): Métrica fundamental para la viabilidad en un entorno de monitoreo real.

## 4. Límites del Alcance (para el MVP)
Quedan excluidos de esta primera entrega:
- Desarrollo de interfaces gráficas de usuario (Frontend web/apps).
- Implementación de bases de datos relacionales o históricos de pacientes.
- Transformación de las señales al dominio tiempo-frecuencia (Espectrogramas / Procesamiento 2D).
- Técnicas de Transfer Learning y preentrenamiento en datasets externos.

El MVP consistirá exclusivamente en un script por línea de comandos que recibe un bloque de datos temporales, realiza la inferencia utilizando la CNN-1D preentrenada, y retorna la clasificación binaria (Crisis / No Crisis).

## 5. Estructura del proyecto

```
ia-final/
├── README.md
├── requirements.txt
├── data/
│   └── processed/               <- split.json, ventanas generadas, stats
├── models/                      <- checkpoints (.pt) del modelo
├── notebooks/                   <- exploración
└── src/
    ├── __init__.py
    ├── config.py                <- configuración centralizada
    ├── annotations.py           <- parser de chbXX-summary.txt (anotaciones)
    ├── preprocessing.py         <- carga EDF, filtrado, ventaneo, etiquetado, escalado robusto (mediana + IQR)
    ├── data.py                  <- Dataset/DataLoader PyTorch
    ├── model.py                 <- arquitectura CNN-1D apilada
    ├── train.py                 <- loop de entrenamiento + métricas + checkpoint
    ├── evaluate.py              <- evaluación inter-paciente (test completo)
    └── inference.py             <- CLI de inferencia sobre .edf
```
