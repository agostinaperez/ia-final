"""
Arquitectura CNN-1D
Red convolucional 1D que clasifica cada ventana de EEG como "crisis" (1) o "no crisis" (0).
6 bloques Conv1D + BatchNorm + ReLU + MaxPool, seguidos de Global Average Pooling, una capa densa y la salida.

    python -m src.model
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchinfo import summary
from src.config import CONV_CHANNELS, CONV_KERNELS, DROPOUT, FC_UNITS, N_CHANNELS

class SeizureCNN(nn.Module):
    """
    CNN-1D para clasificación binaria de ventanas de EEG.

    Entrada:  (batch, N_CHANNELS, WIN_SAMPLES) = (B, 16, 1310). -> el batch es cuántas ventanas le paso
    Salida:   (batch, 1) con el logit de "probabilidad de crisis". -> acá es 1 logit POR BATCH, o sea por ventana

    La arquitectura se arma a partir de las listas de config.py:
      - CONV_CHANNELS = [64, 128, 256, 256, 256, 256]  (filtros por bloque)
      - CONV_KERNELS  = [7, 5, 5, 3, 3, 3]             (tamaño del kernel)
    Cada bloque reduce la longitud temporal a la mitad (MaxPool de 2).
    """

    def __init__(self, in_channels: int = N_CHANNELS, conv_channels: list[int] = CONV_CHANNELS,
                 conv_kernels: list[int] = CONV_KERNELS, fc_units: int = FC_UNITS, dropout: float = DROPOUT):
        # `super().__init__()` registra el módulo en PyTorch para que sus  parámetros y submódulos se rastreen, se muevan a GPU y se guarden.
        super().__init__()

        # tiene que haber un kernel por cada número de filtros (se leen en paralelo, una posición por bloque)
        if len(conv_channels) != len(conv_kernels):
            raise ValueError(
                f"CONV_CHANNELS ({len(conv_channels)}) y CONV_KERNELS "
                f"({len(conv_kernels)}) deben tener la misma longitud."
            )

        # 6 bloques convolucionales, cada uno con 4 capas (Conv1d + BatchNorm + ReLU + MaxPool)
        #la cantidad de kernels decrecen (los primeros bloques ven contexto más genérico y los últimos refinan detalles),
        #y la cantidad de filtros crece (cada grupo de capas combina patrones de las anteriores y necesita más capacidad)
        blocks: list[nn.Module] = []          # acá acumulo los 6 bloques
        c_in = in_channels                    # entrada del primer bloque: 16 canales
        for c_out, kernel in zip(conv_channels, conv_kernels):
            # Agrego un bloque que va de `c_in` canales a `c_out` con kernel `k`
            #O sea el primer bloque recibe 16 canales, le pasa a la próxima como entrada 64 filtros con features, todo con stride 7.
            #el segundo recibe 64 filtros, le pasa a la próxima 128, 
            blocks.append(self._make_block(c_in, c_out, kernel))
            c_in = c_out                      # el output de un bloque es el input del siguiente

        # `nn.Sequential` encadena los bloques en el orden en que los agregué, la salida de uno es la entrada del otro, todo en una sola pasada
        self.conv = nn.Sequential(*blocks)

        # Global Average Pooling (GAP).  Entra (B, 256, 20): 256 canales, cada uno de largo 20.
        # Toma CADA canal por separado y promedia sus 20 muestras en UN solo número (el promedio a lo largo del TIEMPO). Como hay 256 canales, salen 256 promedios -> (B, 256, 1).
        # pytorch elige el tamaño del kernel solo, x eso el "adaptative"
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout después del pooling: apaga neuronas al azar en train (40%) para evitar el overfitting; en eval/inferencia no apaga nada.
        self.drop1 = nn.Dropout(dropout)

        #hago un último bloque, este "fc"-> fully connected -> capas densas.
        #antes de entrar a esta capa yo hago un squeeze para sacar la última dimensión (B, 256, 1)
        #paso de tener algo corte [[[a], [b], ...],[[c], [d], ...],..] a tener [[a, b, ...],[c, d, ...]]
        #o sea no pierdo valores pero sí cambian las dimensiones. Eso es lo q entra a mi capa lineal, y se procesa en paralelo para cada batch
        #o sea recibe una cantidad B de tensores de 256 elementos. Cada uno de esos elementos representa cuánto se activó cada filtro (promedio en el tiempo)
        #la capa linear tiene la matriz de pesos W 256x256 y el vector de biases b de 256 números.
        #cada salida z es la suma ponderada de cada una de las entradas, + el bias
        #z[i][j]=(a[i][0]w[i][0]+...+a[i][255]w[i][255]) + b[j], con i que va de 0 a (B-1) y j q va de 0 a 255 
        self.fc = nn.Sequential(
            nn.Linear(c_in, fc_units),   # como salida mantiene las dimensiones: (B, 256)
            nn.ReLU(),                   # aplica max(0, z). esto tampoco cambia dimensión, solo mete no-linealidad entre las densas
            #pq si no, dos Linear seguidas son equivalentes a una sola linear (no aporta nada extra)
            nn.Dropout(dropout),         # regularización antes de la salida. Aoaga el 40% de esos 256 números (x ventana) y los q sobreviven los multiplica x 1/(1-dropout) para compensar
            nn.Linear(fc_units, 1),      # ahora tengo B tensores, cada uno con un solo elemento q indica si tengo o no crisis
        )

    def _make_block(self, c_in: int, c_out: int, kernel: int) -> nn.Sequential:
        """
        - Conv1d con padding=k//2 (same): relleno con 0s así el largo temporal no cambia en la convolución
          Conv1d aprende `c_out` filtros, cada uno de `c_in . kernel` pesos + el bias (un bias distinto x canal de salida, lo cual me suma +c_out parámetros en el conteo)

              salida = suma(entrada × pesos) + bias

          Sirve para que el filtro se "encienda" (dé un valor distinto de 0) incluso cuando su combinación lineal da 0: es un umbral de encendido aprendible

        - BatchNorm1d normaliza las activaciones por canal (media 0 var 1), restando la media del batch y dividiendo por el desvío, y después re-escala usando 2 parámetros aprendidos por canal (se entrenan como cualquier peso)

              x' = gamma * (x - media)/sqrt(var + eps) + beta

            - gamma (escala): deja que la red "estire" o "comprima" la activación ya normalizada. Sin él, la salida tendría SIEMPRE media 0 y varianza 1, y eso puede no ser lo mejor para la capa que viene.
            - beta (corrimiento): el "bias" del BatchNorm; mueve la activación hacia arriba o abajo.
         
          Son 2 parámetros por canal (por eso BatchNorm1d(c_out) aporta 2*c_out). La normalización en sí estabiliza y acelera el entrenamiento; gamma y beta le devuelven a la red la libertad de
          elegir la escala y posición que más le convenga en cada punto.
    

        - ReLU introduce no-linealidad (sin ella, apilar capas sería lineal).
        - MaxPool1d(2) baja el largo temporal a la mitad, quedándose con el máximo de cada par de muestras: reduce cómputo y hace el modelo más invariante a pequeños corrimientos temporales.
        """
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Toma un batch de ventanas (B, 16, 1310) y devuelve (B, 1) con el logit de "crisis" para cada una.
        """
        x = self.conv(x)        # (B, 16, 1310) -> (B, 256, 20)  [6 bloques]
        x = self.gap(x)         # (B, 256, 20)  -> (B, 256, 1)   [promedia tiempo]
        x = x.squeeze(-1)       # (B, 256, 1)   -> (B, 256)      [quito la dim de 1]
        x = self.drop1(x)       # dropout sobre el vector de 256 features
        x = self.fc(x)          # (B, 256) -> (B, 1)             [logit]
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Igual q forward pero aplicando sigmoide: devuelve la probabilidad de crisis en (0, 1). Se usa en evaluación e inferencia para comparar con el treshold.
        """
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__": #solo corre esto si yo ejecuto el archivo directo. cuando otro archivo lo use esto no se ejecuta
    from src.config import SEED, WIN_SAMPLES
    #prueba para ver si se arma bien y no se rompe cuando le paso datos. no uso mi dataset posta todavía, le paso números random
    torch.manual_seed(SEED)    # fijo el generador de numeros aleatorios de pytorch en 42, para q sea reproducible

    model = SeizureCNN()  #construyo la red q definí arriba! se inicializan los pesos al azar
    # Cuento los parámetros entrenables totales, cosa de ver anomalías (en teoría debería tener entre 500k y 2 millones)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {n_params:,}")

    #hago un batch trucho, randn devuelve el tensor relleno con ruido gaussiano, q son básicamente
    #números random con media 0. En esta prueba me interesa q se banque procesar datos en esa forma, no el contenido de los datos
    x = torch.randn(2, N_CHANNELS, WIN_SAMPLES)

    #acá pongo el flag self.training = false. Las capas leen eso para saber si están siendo usadas o entrenadas
    #todo funciona igual excepto el dropout (en eval no se usa pq solo tiene sentido regularizar si entreno), y el batchnorm
    #use sus estadísticas guardadas en vez de las del batch
    model.eval()
   
   #cosas gráficas que me sirven a mí
    summary(model, input_size=(64, 16, WIN_SAMPLES))
    #a esto lo veo en https://netron.app
    torch.onnx.export(model, x, "model.onnx",
                    input_names=["eeg"], output_names=["logit"])
   
   #desactivo el cálculo de los gradientes, todavía no me hace falta guardar el grafo de derivadas, me ahorra memoria y tiempo
    with torch.no_grad():
        #escribir model(x) es equivalente a escribir model.forward(x). Y eso es, básicamente, meter x a la red y dejar q fluya
        #hasra el otro extremo. Pasa por todas las capas: los 6 bloques convolucionales, el GAP, el dropout, y la capa densa.
        logits = model(x) #devuelve un número por ventana
        proba = model.predict_proba(x) #lo mismo de arriba pero con la sigmoide!! o sea devuelve la probabilidad, pq el sigmoide me da un valor entre 0 y 1

    print(f"Entrada:   {tuple(x.shape)}")
    print(f"Logits:    {tuple(logits.shape)}  -> {logits.flatten().tolist()}")
    print(f"Prob:      {tuple(proba.shape)}  -> {proba.flatten().tolist()}")
    print("Smoke-test de model.py OK.")
