"""
Preprocesamiento de señales EEG de CHB-MIT.

Convierte un archivo `.edf` crudo en ventanas etiquetadas listas para la cnn

    python -m src.preprocessing --compute-stats             # stats del scaler sobre TRAIN
    python -m src.preprocessing --test-file <archivo.edf>   # smoke-test de un solo archivo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import scipy.signal
from sklearn.preprocessing import RobustScaler

from src.annotations import load_annotations
from src.config import (
    CHANNELS_TUEV,
    DATASET_DIR,
    FILTER_ORDER,
    FS,
    HIGH_FREQ,
    LOW_FREQ,
    ROBUST_QUANTILE_RANGE,
    SCALER,
    SCALER_STATS_FILE,
    SPLIT_FILE,
    STATS_SAMPLE_STRIDE,
    STATS_WINDOW_STRIDE,
    STRIDE_SAMPLES,
    WIN_SAMPLES,
    WIN_SECONDS,
)

# En EEG, cada canal mide una diferencia de potencial entre 2 electrodos.
# - Montaje monopolar: cada canal es un electrodo contra una referencia común (ej. "F7-CS2").
# - Montaje bipolar: cada canal es la resta de DOS electrodos vecinos (ej. "FP1-F7").
#  Al restar se anula cualquier señal común a ambos (ruido, derivas, referencia) y queda resaltada la actividad LOCAL entre esa pareja
#en este dataset (en casi todos los archivos) se usan montajes BIPOLARES
VALID_CHANNELS = list(CHANNELS_TUEV)

# lista de todos los electrodos individuales que aparecen en los 16 canales bipolares.
ELECTRODES = sorted({e for j in VALID_CHANNELS for e in j.split("-")})

# Separadores posibles entre electrodos en un label
_SPLIT_RE = re.compile(r"[-–—]")


def normalize_channel_name(name: str) -> str:
    #Normaliza el nombre de un canal para poder compararlo. Pasa a mayúscula, sin espacios, corrige typos (como O1 en vez de 01) y sufijos (en los canales duplicados)
    s = name.strip().upper().replace(" ", "")
    s = s.replace("01", "O1")
    s = re.sub(r"-\d+$", "", s)
    return s

# reconstrucción de los 16 canales válidos
#corre una vez por archivo EDF
def _solve_selection(normalized_edf_channels: list[str]) -> dict | None:
    # bipolares_directos: canales bipolares DIRECTOS que ya vienen armados en el EDF.
    bipolares_directos: dict[str, int] = {}
    # monopolares: canales REFERIDOS a una referencia común.
    monopolares: dict[str, dict[str, int]] = {}
    # unicos: electrodos ÚNICOS (sin referencia visible).
    unicos: dict[str, int] = {}

    for i, label in enumerate(normalized_edf_channels):
        # Separo el nombre por guiones
        parts = [p for p in _SPLIT_RE.split(label) if p]
        if len(parts) == 2:
            a, b = parts
            if a in ELECTRODES and b in ELECTRODES:
                # canal bipolar directo! y la fila en la q está (i)
                bipolares_directos.setdefault(f"{a}-{b}", i)
            elif a in ELECTRODES:
                # "electrodo-REFERENCIA"
                #creo una sub entrada del diccionario monopolares con el nombre de la referencia (b), y adentro guardo
                #todos los electrodos q se comparan contra esa referencia (a) y su fila
                monopolares.setdefault(b, {}).setdefault(a, i)
        elif len(parts) == 1 and parts[0] in ELECTRODES:
            # electrodo único con su fila (i)
            unicos.setdefault(parts[0], i)

    #NO ME DEVUELVE LOS CANALES RECONSTRUIDOS PQ ESTO SE LLAMA EN VARIOS LUGARES
    #PERO SI ME DEVUELVE LAS INSTRUCCIONES DE COMO HACER ESA RECONSTRUCCION, ESO SERÍA MI PLAN
    plan: dict[str, tuple] = {}
    mode = "direct" #valor inicial
    for channel_pair in VALID_CHANNELS:
        #el for corre 16 veces, uno x canal
        # Si el canal q quiero ya existe en este edf como bipolar directo, anoto en el plan q solo tomo su fila y listo.
        if channel_pair in bipolares_directos:
            plan[channel_pair] = ("row", bipolares_directos[channel_pair])
            continue
        
        # sino, se que para este EDF, a este canal lo reconstruyo por resta
        a, b = channel_pair.split("-")
        #en la primer pasada entro acá, y defino q estructura tienen los canales a reconstruir de este archivo
        if mode == "direct":
            common_refs = [r for r in monopolares if a in monopolares[r] and b in monopolares[r]]
            if a in unicos and b in unicos:        #electrodos únicos
                mode = "single"
            elif common_refs:                        # ref común (ej. CS2)
                mode = f"ref:{common_refs[0]}"
            else:
                return None                          # no hay forma de armar el par

        if mode == "single":
            # Resto: A - B usando los electrodos únicos.
            if a not in unicos or b not in unicos:
                return None
            plan[channel_pair] = ("diff", unicos[a], unicos[b])
        else:
            # Modo "ref:CS2": resto dos canales que comparten la MISMA ref.
            # FP1-F7 = (FP1-CS2) - (F7-CS2). La CS2 se cancela → queda FP1-F7.
            ref = mode.split(":", 1)[1]
            if ref not in monopolares or a not in monopolares[ref] or b not in monopolares[ref]:
                return None
            plan[channel_pair] = ("diff", monopolares[ref][a], monopolares[ref][b])

    return plan


def load_16_channels(path, fs_expected: int = FS) -> tuple[np.ndarray | None, str | None]:
    """
    Carga el EDF con MNE y devuelve la señal de los 16 canales válidos en orden canónico
    (filas), reconstruyendo por resta los que haga falta
    Devuelve (data, reason):
      - data: (16, N) float64, reason None, plan completo del montaje.
      - data None, reason != None y plan=None si el archivo hay que descartarlo.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])
    if abs(sampling_rate - fs_expected) > 1e-6:
        return None, f"sampling_rate_invalida ({sampling_rate:g} Hz != {fs_expected})"

    normalized_edf_channels = [normalize_channel_name(c) for c in raw.ch_names]
    plan = _solve_selection(normalized_edf_channels)
    if plan is None:
        return None, "channels_incompatible"
    
    #acá ejecuto el dichoso plan! 
    #esto es una matriz, en donde cada fila es un canal, y c columna es el voltaje en un instante de tiempo
    # o sea una matriz de, si no me equivoco, 16x921600 (porque mi frec muesteo es 256 hz, y tengo 1 hora, así q 3600*256)
    data = raw.get_data()
    
    #yo ahora creo OTRA matriz c 16 filas, y cantidad de columnas fijas
    out = np.empty((len(VALID_CHANNELS), data.shape[1]), dtype=np.float64)
    #después, eventualmente, a eso lo corto en ventanas de 5.12 segundos. No acá
    #agora sim reconstruyo. k es mi fila. 
    for k, channel_pair in enumerate(VALID_CHANNELS):
        #cada fila de out (mi matriz limpita) sale de copiar una fila de data o de restar 2
        op = plan[channel_pair]  
        #nunca modifico las filas de data per se, sino que las uso para construir las filas de out
        if op[0] == "row":
            out[k] = data[op[1]] #solo copio la fila entera
        else:
            # Resta de dos canales que comparten referencia:
            #   (A - REF) - (B - REF) = A - B
            # La referencia se cancela, queda el canal bipolar puro
            #LA RESTA ES DE LA FILA COMPLETA, X ENDE ELEMENTO A ELEMENTO
            out[k] = data[op[1]] - data[op[2]]
    return out, None


# Filtrado, ventaneo y etiquetado
def filter_bandpass(data: np.ndarray, fs: int = FS, low: float = LOW_FREQ, high: float = HIGH_FREQ, order: int = FILTER_ORDER) -> np.ndarray:
    """
    Butterworth pasa-banda con fase 0 para evitar distorsion temporal. Filtra a lo largo del eje 1  de cada canal.
    """
    
    #construyo la heramienta matemática para filtrar. especifico que es de tipo "bandpass",
    #pido q me lo devuelva en formato "sos" (second order sections) para que no me devuelva una única ecuación gigante.
    #pq, al final del día, el filtro es un polinomio de orden 5 en mi caso, entonces se puede hacer lío con los decimales
    #lo q hace básicamente el formato sos es q me divide mi ecuación de orden 5 en una de orden 2, desp otra de orden 2, y una de prden 1
    #Eentonces aplica filtritos seguidos
    butterworth_filter = scipy.signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    #acá agarro el filtro y se lo paso a mis datos en la dimensión 1 (el del tiempo, pq la dimensión 0 son los canales)
    #ese sosfiltfitl básicmanete hace que sea de "fase 0", porque filtra de izq a derecha y desp de derecha a izq cosa de anular
    #cualquier desfase temporal
    return scipy.signal.sosfiltfilt(butterworth_filter, data, axis=1)


def segment_windows(data: np.ndarray, win_samples: int = WIN_SAMPLES, stride: int = STRIDE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """
    Parte una señal (C, N) en ventanas solapadas de (n_windows, C, win_samples).

    Devuelve (windows float32, indices_inicio en muestras).
    Sólo se emiten ventanas COMPLETAS: la sobra al final se descarta.
    """
    n = data.shape[1] #me devuelve la cantidad de elementos que tengo en la dimensión temporal
    
    #calculo la cantidad de ventanas que tengo en este archivo. Tomo todas mis muestras, les resto la cantidad total de muestras por ventana (de ahí ya saco mi primer ventana, x eso el +1)
    # y hago una división entera (//) considerando el 50% overlap, para saber cuantos "pasos" hacia la próxima ventana voy a tener. de ahí saco mi canridad de ventanas
    n_windows = (n - win_samples) // stride + 1
    if n_windows <= 0: #No puedo tener 0 ventanas, ni cantidad negativa. Devuelvo matrices vacías.
        return np.empty((0, data.shape[0], win_samples), dtype=np.float32), np.empty(0, dtype=np.int64)

    # crea ventanas contiguas sin guardarlas en la memoria RAM, pero NO TOMA MI STRIDE TODAVÍA, por default va con stride 1
    #me devuelve una matriz (numpy.ndarray) con 3 dimensiones! La dimenisón 0 es los canales, la 1 es lasventanas q pudo armar, la 2 es el tamaño x ventana
    # o sea para el canal C tendría una cantidad de ventanas W, y cada ventana es un array de WS elementos
    view = np.lib.stride_tricks.sliding_window_view(data, win_samples, axis=1)  # el shape me da la tupla (Cantidad canales, cant ventanas, win_samples)
    
    #ahora sí aplico mi stride y hago el guardado físico en la memoria RAM. 
    #adentro de view tengo tres comas, cada una da una instrucción a una dimensión de la matriz view. En la dimensión 0 y la 2 pongo :, o sea digo q me de todo y sin recortar!
    #en la 1 hago ::stride -> la regla es escribir inicio:fin:paso . Así q mantengo el inicio y el fin, pero aplico el solapamiento que yo definí
    windows = np.ascontiguousarray(view[:, ::stride, :])  # (C, W, WS)
    #ahora acomodo las dimensiones de la matriz, porque pytorch exige que el lote de muestras sea la 1er dimensión (la 0)
    #guardo los decimales a 32 bits (estaban en 64) para q los datos pesen menos y se acelere el entrenamiento (no me hace falta tener 64 bits)
    windows = np.transpose(windows, (1, 0, 2)).astype(np.float32)
    #creo el "anotador". Agarro la cantidad de ventanas q voy a tener (ej 10), lo hago array (0, 1, 2, 3,...) y multiplico x mi stride (655), entonces queda (0, 655, 1310,...) y sé donde arranca cada ventana
    starts = np.arange(n_windows, dtype=np.int64) * stride
    return windows, starts


def label_windows(windows_starts: np.ndarray, n_win: int, seizures: list[tuple[int, int]], fs: int = FS, win_seconds: float = WIN_SECONDS) -> np.ndarray:
    """
    Etiqueta binaria: una ventana es CRISIS (1) si solapa cualquier porción de un intervalo de crisis anotado; si no, NO CRISIS (0).
    """
    labels = np.zeros(n_win, dtype=np.int8) #creo un arreglo lleno de 0s
    if not seizures:
        return labels
    
    start_seconds = windows_starts / fs
    end_seconds = start_seconds + win_seconds
    #uso indexado con máscaras booleanas. Yo evalúo a la vez todos los elementos de los array start seconds y end seconds, y creo una mask booleana llamada overlap,
    #que termina siendo un array del mismo tamaño que contiene [True, False, False, etc]
    #entonces, para cada una de las crisis (a esas si las recorro iterativamente), las comparo en bloque contra todas las ventanas, y para cada crisis obtengo un array de booleanos
    for (start_seizure, end_seizure) in seizures:
        #ej para q se vea más claro. una ventana de 0s a 5.12 s, un seizure q arranca en segundo 4 y termina en segundo 7.
        # overlap = (0<7) & (5.12 > 4) -> ambas se cumplen! va el overlap
        overlap = (start_seconds < end_seizure) & (end_seconds > start_seizure)
        #acá aplico la máscara. "superpongo" el array de labels con el de overlap (tienen el mismo tamaño) y, en donde dice True, sobreescribo un 1 
        labels[overlap] = 1
    return labels


# Escalado robusto por canal
def preprocess_train_data(files: list[Path], seizures_by_file: dict[str, list[tuple[int, int]]],
                         scaler: str = SCALER, verbose: bool = True) -> dict:
    """
    Recorre los archivos indicados, arma las ventanas filtradas y estima las estadísticas del RobustScaler por canal sobre TODAS esas ventanas:
    Devuelve un diccionario con las stats + el detalle de cuántos archivos/ventanas/puntos se usaron para estimar los cuantiles.
    """

    # guardar todas las muestras temporales de train es inviable, así q acumulo una VERSIÓN SUBMUESTREADA de las ventanas:
    #   - 1 de cada STATS_WINDOW_STRIDE ventanas,
    #   - dentro de cada una, 1 de cada STATS_SAMPLE_STRIDE muestras.
    # Con 1/16 x 1/16 quedan ~55 M de valores (~220 MB en float32) y los cuantiles convergen al valor exacto del train completo: la mediana y el IQR son estadísticos robustos y un submuestreo uniforme no los sesga.
    bloques: list[np.ndarray] = []  # un bloque (n_muestras_sub, 16 canales) por archivo
    ok_files_counter = 0 #contadores de auditoria
    number_of_windows = 0
    number_of_samples = 0
    number_of_data_points = 0
    
    # random number generator con una semilla fija para q sea reproducible!
    rng = np.random.default_rng(0)
    
    #arranco el ciclo para leer los EEG de los pacientes uno por uno. Me fijo si tiene crisis
    for path in files:
        seizures = seizures_by_file.get(path.name, [])
        #preproceso mis datos (abro el archivo, divido en ventanas, NO ESCALO TODAVÍA)
        res = process_edf(path, seizures, do_scale=False)
        if not res["ok"]:
            if verbose:
                print(f"   [skip] {path.parent.name}/{path.name}: {res['reason']}")
            continue
        
        windows = res["windows"]  # (n_ventanas, 16 canales, 1310 muestras)
        # Submuestreo aleatorio: 1 de cada STATS_WINDOW_STRIDE ventanas al azar; dentro de cada ventana, 1 de cada STATS_SAMPLE_STRIDE muestras al azar (no es estratificado igual,
        # profe si llega a estar leyendo esto, no sé si debería hacerlo estratificado o no pero me parece que con que sea proporcionalmente 1/16 muestras al azar estoy bien)
        # Un muestreo uniforme aleatorio no sesga los cuantiles (mediana/IQR) pq es robusto!
        
        n_kept = max(1, windows.shape[0] // STATS_WINDOW_STRIDE) #cuantas ventanas voy a tomar para el muestreo
        #uso mi random number generator y le paso: mis ventanas, el tamaño total del array (la cantidad de ventanas q voy a muestrear), replace False para q no se pueda elegir el mismo valor varias veces
        #esto me devuelve un array con el número de ventanas seleccionadas (por ejemplo [2, 18, 35...qsy])
        kept_idx = rng.choice(windows.shape[0], size=n_kept, replace=False)
        
        n_off = max(1, WIN_SAMPLES // STATS_SAMPLE_STRIDE)  #1310/16 = 81 pq es división entera. cantidad de elementos x ventana q voy a tomar para el muestreo
        #idem q arriba, offsets para a ser un array de 81 elementos, con números random seleccionados que van de 0 a 1310
        offsets = rng.choice(windows.shape[2], size=n_off, replace=False)
        
        # windows[kept_idx] me descarta todas las ventanas q no salieron elegidas. Quedo con dimensiones (n_kept, 16, 1310)
        # con [:, :, offsets], aplico una transformación sobre las 3 dimensiones. La 0 y la 1 las dejo igual (:), pero a la 2 (cantidad de elementos x ventana)
        # sí la transformo, tomando los elementos q estén en los índices rarndom que salieron en offsets. Esto se aplica en todos los canales y todas las ventanas obvio
        w_sub = windows[kept_idx][:, :, offsets] #paso a tener dimensiones (n_kept, 16, 81)
        #acá tengo q hacer una reestructuración de mis datos pq el robustScaler me pide una tabla 2D tradicional, en donde cada canal sea una columna
        #con el transpose paso a (ventanas, tiempo, canales). A eso lo guardo en la RAM como un array contiguo en memoria, y le hago el reshape (-1, 16). Exijo 16 columnas,
        # y ese -1 es decir q se calcule automáticamente la cantidad de filas q hacen falta. Lo q hago es "aplastar" las primeras 2 dimensiones. Agarro todas las ventanas y todos
        #los instantes de tiempo y los fundo en una sola lista. Me queda fila 0: ventana 0 valores en tiempo 0. Fila 1 ventana 0 valores en tiempo 1, ..., fila X ventana 1 valores en tiempo 0, y así
        w_sub = np.ascontiguousarray(w_sub.transpose(0, 2, 1)).reshape(-1, len(VALID_CHANNELS))
        #guardo esa tabla 2D (corresponde a un solo archivo de un solo paciente) adentro de la lista bloques
        bloques.append(w_sub)
        #actualizo mis contadores de auditoría!
        number_of_windows += windows.shape[0]
        number_of_samples += windows.shape[0] * windows.shape[2]
        number_of_data_points += w_sub.shape[0]
        ok_files_counter += 1
        if verbose: #por default es true pq quiero ver q anda pasando
            print(f"   [ok]   {path.parent.name}/{path.name}: {windows.shape[0]} ventanas, "
                  f"{int(res['labels'].sum())} positivas")

    if ok_files_counter == 0 or not bloques:
        raise RuntimeError("No se procesó ningún archivo: no hay stats que computar.")

    #ahora sí puedo calcular la mediana y el IQR!! Bloques es una lista de tablas 2d, una por cada archivo del subconjunto de train.
    #con el concatenate en axis 0 agarro todas las tablas separadas y las apilo una abajo de la otra en una tabla gigante. Sigue teniendo 16 columnas, 1 x canal
    X = np.concatenate(bloques, axis=0)
    
    # RobustScaler de scikit-learn calcula la MEDIANA (center_) y el IQR = Q75 - Q25 (scale_) por canal. Me devuelve una matriz con 2 vectores de 16 elementos
    robust = RobustScaler(quantile_range=ROBUST_QUANTILE_RANGE).fit(X)
    del bloques, X  # libero las matrices gigantes y los bloques de la memoria

    stats = {
        "scaler": "robust",
        "channels": VALID_CHANNELS,
        "n_files": ok_files_counter,
        "n_windows": number_of_windows,
        "n_samples": number_of_samples,
        "n_points": number_of_data_points,
        "median": robust.center_,  # mediana por canal
        "iqr": robust.scale_,      # rango intercuartílico (Q75 - Q25) por canal
    }
    return stats


def save_stats(stats: dict, out: Path = SCALER_STATS_FILE) -> Path:
    """Persiste las stats del escalador robusto en un .npz (mediana + IQR por canal)."""
    #el npz es el formato comprimido oficial de numpy y está bueno porque no pierde la precisión de los decimales!!
    #aparte se inyecta rápido en la memoria RAM para cuando lo esté leyendo la CNN
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "scaler": np.array([stats["scaler"]]),
        "channels": np.array(stats["channels"], dtype=np.str_),
        "n_files": np.array([stats["n_files"]]),
        "n_windows": np.array([stats["n_windows"]]),
        "n_samples": np.array([stats["n_samples"]]),
        "n_points": np.array([stats["n_points"]]),  # puntos (submuestreados) usados para los cuantiles
        "median": stats["median"],                   # mediana por canal (center_ del RobustScaler)
        "iqr": stats["iqr"],                         # IQR (Q75 - Q25) por canal (scale_ del RobustScaler)
    }
    np.savez_compressed(out, **arrays)
    return out


def load_scaler_stats(path: Path = SCALER_STATS_FILE) -> dict | None:
    """Carga las stats del escalador robusto. Devuelve None si no existen todavía."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        scaler_name = str(z["scaler"][0])
        if scaler_name != "robust":
            # Si el npz quedó de una corrida vieja (z-score/min-max), avisamos y
            # pedimos regenerarlo en vez de cargar stats que ya no corresponden.
            raise RuntimeError(
                f"scaler_stats.npz contiene el escalador '{scaler_name}' y el "
                "proyecto usa SOLO RobustScaler. Regeneralo con: "
                "python -m src.preprocessing --compute-stats"
            )
        stats = {
            "scaler": scaler_name,
            "channels": [str(c) for c in z["channels"]],
            "n_files": int(z["n_files"][0]),
            "n_windows": int(z["n_windows"][0]),
            "n_samples": int(z["n_samples"][0]),
            "n_points": int(z["n_points"][0]) if "n_points" in z.files else 0,
            "median": z["median"],  # mediana por canal (center_ del RobustScaler)
            "iqr": z["iqr"],        # IQR (Q75 - Q25) por canal (scale_ del RobustScaler)
        }
    return stats


def apply_scaler(windows: np.ndarray, stats: dict | None) -> np.ndarray:
    """
    escalado robusto por canal usando las stats: x' = (x - mediana_c) / IQR_c
    Matemáticamente idéntico a RobustScaler.transform() de scikit-learn usando las stats guardadas en el npz. Si stats es None (no hay stats o no se pidió),
    devuelve las ventanas sin tocar.
    """
    if stats is None or windows.shape[0] == 0:
        return windows
    
    w = windows.astype(np.float64) #copia mi arreglo y cambia el tipo de datos a 64 bits para q la resta y división sea precisa. Después lo vuelvo a 32 por los motivmos mencionados en otras funciones
    
    
    median = stats["median"][:, None]  # median es un vector de 16 números (una mediana x canal), están ordenadas en en orden de los canales. Le añado una nueva dimensión
    #así median queda de (16, 1) y se alínea (desde la derecha) con w que es (numero_de_ventanas, 16, 1310). Los 16 quedan alineaditos
    iqr = stats["iqr"][:, None]        # IQR por canal (hago lo mismo q hice con median recién)
    
    denom = np.where(iqr == 0, 1.0, iqr)#evito dividir por 0 entonces si en algún canal x error quedó el iqr 0 (sería q tuvo valores ctes todo el tiempo), lo seteo en 1

    # QUÉ PASA ACÁ (broadcasting), con un ejemplo pq me pongo gagá y me cuesta seguirlo. Los array de numpy son muy exóticos
    #
    #   w      tiene forma (n_ventanas, n_canales, n_muestras).
    #   median tiene forma (n_canales, 1), denom  tiene forma (n_canales, 1)
    #
    # Numpy alinea las formas DESDE LA DERECHA y "estira" lo que haga falta:
    #
    #   w      : (n_ventanas, n_canales, n_muestras)
    #   median : (           , n_canales,          1)
    #              -> dim muestras : n_muestras  vs  1  -> se estira a n_muestras
    #              -> dim canales  : n_canales   vs  n_canales -> coinciden
    #              -> dim ventanas : n_ventanas  vs  (nada) -> se estira
    #   resultado: (n_ventanas, n_canales, n_muestras)  (misma forma q w)
    #
    # ejemplo concreto (2 ventanas, 3 canales, 3 muestras):
    #
    #   w = [                        median (3,1) = [[20],   -> canal 0
    #  canal1 [[10, 32, 15],                       [ 8],   -> canal 1
    #  canal2 [ 4,  8, 20],                       [ 3]]   -> canal 2
    #         [ 1,  2,  3]],
    #(Ventana2:)
    #        [[30, 12, 40],
    #         [ 9, 21, 17],
    #         [ 7,  5,  6]]]
    #
    #   (w - median) resta el escalar de cada canal a TODAS las muestras de TODAS las ventanas de ese canal (median[0]=20 se resta a todo el canal 0,
    #   median[1]=8 a todo el canal 1, etc.), muestra por muestra:
    #
    #        canal 0, ventana 0: [10-20, 32-20, 15-20] = [-10, 12, -5]
    #        canal 0, ventana 1: [30-20, 12-20, 40-20] = [ 10, -8, 20]
    #        canal 1, ventana 0: [ 4- 8,  8- 8, 20- 8] = [ -4,  0, 12]
    #        ... y así con cada canal y cada ventana.
    #
    #   Después se divide cada valor por denom[canal] (el IQR de ESE canal), y listo: cada canal queda centrado y re-escalado por SU propia mediana e IQR, sin que un canal se mezcle con otro.
    out = (w - median) / denom
    return out.astype(np.float32)


# Pipeline completo de un archivo
#por default es true porq eventualmente en prod lo voy a llamar con true, pero por ahora acá no
def process_edf(path, seizures: list[tuple[int, int]], *, fs: int = FS, win_samples: int = WIN_SAMPLES, stride: int = STRIDE_SAMPLES,
                scaler_stats: dict | None = None, do_scale: bool = True) -> dict:
    """
    Procesa UN archivo EDF completo:
    EDF -> 16 canales TUEV -> filtro 0.5–50 Hz -> ventanas 5.12 s / 50% -> etiquetado -> (si scale es true) escala por canal con stats de train.

    Devuelve un dict:
        ok=True:  windows (n,W), labels (n), starts (n, seg), fs.
        ok=False: reason (string) y nada más.
    """
    if not Path(path).exists():
        return {"ok": False, "reason": "archivo_no_existe"}

    #matriz_señales es la matriz en la que cada fila es un canal (Ya reconstruido) y c columna contiene la amplitud
    #de la señal en cada instante de tiempo
    matriz_señales, reason = load_16_channels(path, fs_expected=fs)
    if matriz_señales is None:
        return {"ok": False, "reason": reason}

    #hago el butterworth
    filtered = filter_bandpass(matriz_señales, fs=fs)
    #filtro las ventanas
    windows, starts = segment_windows(filtered, win_samples=win_samples, stride=stride)
    #obtengo mi cantidad de ventanas (ahora es mi dimensión 0 después del transpose q hice)
    n_win = windows.shape[0]
    #hago una clasificación binaria de las ventanas del archivo. 1 si se superpone con algún instante de crisis, 0 si no.
    labels = label_windows(starts, n_win, seizures, fs=fs)
    
    if do_scale:
        windows = apply_scaler(windows, scaler_stats)
    #devuelvo toda la datita
    return {
        "ok": True,
        "file": Path(path).name,
        "windows": windows,
        "labels": labels,
        "starts": (starts / fs).astype(np.float64),
        "n_windows": n_win,
        "n_positive": int(labels.sum()),
        "fs": fs,
    }

# HELPERS DEL CLI---------------------------------------------------------------------

def _train_files_from_split(data_dir: Path, split: dict) -> list[Path]:
    """Lista de EDFs (solo los anotados en el summary) para los pacientes de train."""
    files: list[Path] = []
    for patient in split["train"]:
        summary = data_dir / patient / f"{patient}-summary.txt"
        if not summary.exists():
            print(f"[WARN] {patient}: no tiene summary; se omite.")
            continue
        from src.annotations import parse_summary
        for fname in parse_summary(summary):
            p = data_dir / patient / fname
            if p.exists():
                files.append(p)
            else:
                print(f"[WARN] {patient}/{fname} anotado pero no existe en disco.")
    return files


def compute_stats(data_dir: Path, split_file: Path, limit: int | None, patients: list[str] | None) -> None:
    split = json.loads(split_file.read_text(encoding="utf-8"))
    annotations = load_annotations(data_dir)

    train_files = _train_files_from_split(data_dir, split)
    
    #por si quisiera algún paciente en particular o alguna cantidad de archivos como límite en particular. por lo pronto no lo uso
    #pero capaz cuadno haga las próximas fases sí
    if patients:
        wanted = set(patients)
        train_files = [p for p in train_files if p.parent.name in wanted]
        if not train_files:
            sys.exit("Ningún archivo de los pacientes pedidos quedó en TRAIN.")
    if limit:
        train_files = train_files[:limit]

    files_by_name = {p.name: p for p in train_files}
    seizures_by_file = {}
    for patient, files in annotations.items(): #annotations me trae paciente: {nombre de archivo: intervalos de crisis}
        for fname, seizure_intervals in files.items(): #cada intervalo es una tupla de inicio en segundos y fin en segundos
            if fname in files_by_name: #annotations tiene data de pacientes q no están en train, entonces los salteo
                seizures_by_file[fname] = seizure_intervals

    print(f"\nComputando stats del scaler (scaler='{SCALER}') sobre "
          f"{len(train_files)} archivos de TRAIN...")
    
    stats = preprocess_train_data(train_files, seizures_by_file)
    save_stats(stats)
    
    print(f"Stats guardadas en: {SCALER_STATS_FILE}")
    print("\nmediana por canal (center_):", np.array2string(stats["median"], precision=3, suppress_small=False))
    print("IQR por canal (scale_):     ", np.array2string(stats["iqr"], precision=3, suppress_small=False))
    print(f"Ventanas de train usadas: {stats['n_windows']} | archivos: {stats['n_files']}")
    print(f"(mediana e IQR estimados sobre {stats['n_points']} puntos submuestreados)")

def test_file_cli(data_dir: Path, edf_path: str) -> None:
    """CLI --test-file: procesa un único EDF y muestra un resumen (smoke-test)."""
    path = Path(edf_path)
    if not path.exists():
        sys.exit(f"No existe: {path}")

    # Anotaciones del paciente si están disponibles (para etiquetar).
    seizures: list[tuple[int, int]] = []
    patient = path.parent.name if path.parent.name.startswith("chb") else None
    if patient:
        annotations = load_annotations(data_dir)
        seizures = annotations.get(patient, {}).get(path.name, [])

    print(f"\nProcesando {path} ({len(seizures)} crisis anotadas)...")
   
    res = process_edf(path, seizures, do_scale=False)
    if not res["ok"]:
        sys.exit(f"[FAIL] {res['reason']}")
    windows, labels, starts = res["windows"], res["labels"], res["starts"]
    print(f"Ventanas: {windows.shape}  (n, canales, muestras)")
    print(f"Labels: {labels.sum()} positivas de {len(labels)} "
          f"({100 * labels.mean():.2f}%)")
    print(f"Primera ventana positiva a t={starts[labels == 1][0]:.2f} s" if labels.any() else "Sin ventanas positivas")
    # Control del escalado robusto: aplicamos las stats de TRAIN y chequeamos que
    # cada canal quede centrado en mediana~0 con IQR~1 (lo que garantiza el RobustScaler).
    print("\nControl del escalado robusto por canal (tras stats de TRAIN: mediana~0, IQR~1):")
    try:
        stats = load_scaler_stats()
    except RuntimeError as exc:
        # El npz viejo es de z-score/min-max: se avisa y se sigue sin el control.
        print(f"   (aviso: {exc})")
        stats = None
    if stats is None:
        print("   (no se pudo aplicar el control: falta scaler_stats.npz en versión robust)")
    else:
        scaled = apply_scaler(windows, stats)
        for i, ch in enumerate(VALID_CHANNELS[:4]):
            q1, q3 = np.percentile(scaled[:, i, :], [25, 75])
            print(f"   {ch:<8} mediana={np.median(scaled[:, i, :]): .2e}  IQR={q3 - q1: .2e}")


def main() -> None:
    #creo el parser y los argumentos q puede llegar a recibir
    parser = argparse.ArgumentParser(description="Preprocesamiento de EDFs CHB-MIT")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N archivos (smoke-test).")
    parser.add_argument("--patients", type=str, default=None, help="Solo estos pacientes (csv).")

    # acciones posibles. Solo puedo elegir una
    group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--report", action="store_true", help="Escaneo de header de todos los EDF.")
    group.add_argument("--compute-stats", action="store_true", help="Stats de la estandarización.")
    group.add_argument("--test-file", type=str, help="Procesar un único EDF de prueba.")

    # lee sys.argv, chequea que los flags sean válidos
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"--data-dir no es válido: {data_dir}")

    if args.compute_stats:
        patients = args.patients.split(",") if args.patients else None
        compute_stats(data_dir, Path(args.split_file), args.limit, patients)

    #test para un solo archivo (no es para prod)
    else: 
        test_file_cli(data_dir, args.test_file)


if __name__ == "__main__":
    main()