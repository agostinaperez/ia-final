"""
Dataset / DataLoader de entrenamiento.
Entrega a la red las ventanas de EEG "listas" (16 canales, 1310 muestras x ventana, filtradas y escaladas). Las
ventanas se generan en el momento en que se necesitan, no se guardan en el disco, porque sino son 47GB de almacenamiento necesario extra

Ideas clave:
  - Índice global: tabla que me dice a qué archivo pertenece una ventana, y qué número de ventana local tiene dentro de dicho archivo. Tamb si tiene crisis o no. Se arma UNA vez con una lectura de cabeceras.
  - La señal recién se procesa (process_edf, ~1 s/archivo) cuando el DataLoader pide la primera ventana de ese archivo. Como process_edf
    devuelve TODAS las ventanas del archivo de una, se guarda en RAM el ÚLTIMO archivo procesado (caché de 1 archivo) y de ahí se sirven sus ventanas de a una.
  - Train: undersampling controlado (todas las positivas + NEG_POS_RATIO x positivas de negativas al azar) con SHUFFLE GLOBAL. Para poder
    shufflear ventana por ventana SIN reprocesar archivos, cada archivo se procesa una sola vez por época y sus ventanas elegidas se juntan (~1.2 GB) y se shufflean completas. 

    python -m src.data --smoke              # smoke-test local (pocos archivos)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.annotations import load_annotations, parse_summary
from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    FS,
    N_CHANNELS,
    NEG_POS_RATIO,
    NUM_WORKERS,
    SCALER_STATS_FILE,
    SEED,
    SPLIT_FILE,
    STRIDE_SAMPLES,
    WIN_SAMPLES,
)

from src.preprocessing import _solve_selection, label_windows, load_scaler_stats, normalize_channel_name, process_edf

# Armado de listas de archivos
def files_for_patients(data_dir: Path | str, patients: list[str]) -> list[Path]:
    """EDFs anotados (los del summary) para una lista de pacientes"""
    data_dir = Path(data_dir)
    files: list[Path] = []
    for patient in patients:
        summary = data_dir / patient / f"{patient}-summary.txt"
        if not summary.exists():
            print(f"[WARN] {patient}: no tiene summary; se omite.")
            continue
        for fname in parse_summary(summary):
            path = data_dir / patient / fname
            if path.exists():
                files.append(path)
            else:
                print(f"[WARN] {patient}/{fname} anotado pero no existe en disco.")
    return files


def annotations_by_name(annotations: dict) -> dict[str, list[tuple[int, int]]]:
    """Aplana {paciente: {archivo: [(inicio, fin), ...]}} -> {archivo: [...]}."""
    return {fname: ivs for files in annotations.values() for fname, ivs in files.items()}


# Dataset con ventanas + caché de 1 archivo + índice global!
#HEREDO DE DATASET, LA CLASE DE PYTORCH, ASÍ Q TENGO Q DEFINIR __len__ (cuántos items tengo) y __getitem__ (como conseguir item x)
class WindowDataset(Dataset):
    """
    Dataset de ventanas de EEG generadas en el momento.

    - `files`: lista de EDFs (p. ej. todos los de TRAIN).
    - `annotations`: {nombre de archivo: [(inicio, fin), ...]} en segundos.
    - `scaler_stats`: stats del train (de robustscaler). si le paso none = ventanas sin escalar (solo smoke-test).

    El índice global se arma con cabeceras únicamente; la señal de cada archivo se procesa recién cuando hace falta y se guarda
    solo el último archivo procesado para servir sus ventanas de a una.
    """

    def __init__(self, files, annotations, *, scaler_stats=None):
        self.files = [Path(f) for f in files]

        # olo guardo LA REFERENCIA (el "puntero") para poder consultarlo después cuando etiquete ventanas.
        self.annotations = annotations
        # También solo guardo la referencia.
        self.scaler_stats = scaler_stats

        # CACHÉ de 1 archivo: .
        #   _current_path -> ruta del ÚLTIMO archivo que procesé (None = ninguno)
        #   _current      -> el resultado de process_edf de ese archivo (None)
        # Empiezan en None porque todavía no procesé ningún archivo
        self._current_path: Path | None = None
        self._current: dict | None = None

        # ÍNDICE GLOBAL: Se llena UNA sola vez
        #   _valid_files -> lista de archivos válidos
        #   _file_ids    -> por cada ventana global, el NÚMERO de su archivo
        #   _local_ids   -> por cada ventana global, QUÉ ventana local es
        #   _labels      -> por cada ventana global, 1 (crisis) o 0 (no crisis)
        self._built = False
        self._valid_files: list[Path] = []
        self._file_ids: np.ndarray | None = None
        self._local_ids: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    # Índice global (solo cabeceras, sin cargar la señal)
    def _scan_header(self, path: Path) -> tuple[int, np.ndarray] | None:
        """(n_ventanas, labels) leyendo solo la cabecera del EDF."""
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if abs(fs - FS) > 1e-6:
            return None
        names = [normalize_channel_name(c) for c in raw.ch_names]
        if _solve_selection(names) is None:
            return None
        n_total = int(raw.n_times)
        n_windows = (n_total - WIN_SAMPLES) // STRIDE_SAMPLES + 1
        if n_windows <= 0:
            return None
        
        # starts = arange(n_windows) * stride  (como en segment_windows)
        starts = np.arange(n_windows, dtype=np.int64) * STRIDE_SAMPLES
        labels = label_windows(starts, n_windows, self.annotations.get(path.name, []), fs=FS)
        return n_windows, labels

    def _ensure_index(self) -> None:
        """Construye el índice global ventana -> (archivo, ventana local) y el vector de labels"""
        if self._built:
            return
        file_ids, local_ids, all_labels, valid_files = [], [], [], []
        for path in self.files:
            scan = self._scan_header(path)
            if scan is None:
                continue
            n_windows, labels = scan
            file_id = len(valid_files)
            valid_files.append(path)
            file_ids.append(np.full(n_windows, file_id, dtype=np.int32))
            local_ids.append(np.arange(n_windows, dtype=np.int32))
            all_labels.append(labels)
        self._file_ids = np.concatenate(file_ids) if file_ids else np.empty(0, np.int32)
        self._local_ids = np.concatenate(local_ids) if local_ids else np.empty(0, np.int32)
        self._labels = np.concatenate(all_labels).astype(np.int8) if all_labels else np.empty(0, np.int8)
        self._valid_files = valid_files
        self._built = True

    # Caché de 1 archivo procesado
    def _get_processed(self, path: Path) -> dict:
        """Resultado de process_edf para `path`, reutilizando el último archivo procesado si coincide """
        if path == self._current_path:
            return self._current

        res = process_edf(path, self.annotations.get(path.name, []), scaler_stats=self.scaler_stats)
        if not res["ok"]:
            raise RuntimeError(f"[data.py] {path.name} falló dentro del Dataset: {res['reason']}")

        self._current_path = path
        self._current = res
        return res

    # Interfaz de Dataset
    def __len__(self) -> int:
        self._ensure_index()
        return len(self._file_ids)

    def __getitem__(self, idx: int):
        # lo llama el DataLoader de PyTorch, cuando itero `for x, y in loader:`.
        # Por cada índice `idx` que el DataLoader elige, ejecuta `dataset[idx]`,
        # que Python traduce a `dataset.__getitem__(idx)`. 
        
        # me aseguro de tener el índice global armado
        self._ensure_index()

        # El índice global me dice, para esta ventana `idx`:
        #   file_id = el NÚMERO de archivo (su posición en _valid_files)
        #   local_id = la ventana LOCAL dentro de ese archivo
        # Ej: idx=1234 -> file_id=2 (el tercer archivo válido), local_id=200 (su ventana 200).
        file_id = int(self._file_ids[idx])
        local_id = int(self._local_ids[idx])

        # Con el número de archivo, busco su RUTA real en la lista _valid_files.
        path = self._valid_files[file_id]

        # Pido el resultado procesado de ese archivo. Si es el mismo archivo que
        # ya tengo en caché, joia; si no, llama a process_edf y lo guarda en caché.
        res = self._get_processed(path)

        # `res["windows"]` tiene forma (n_ventanas_del_archivo, 16, 1310).
        # Con `[local_id]` RECORTO una sola ventana: la fila número `local_id`, que queda de forma (16, 1310) en float32.
        window = res["windows"][local_id]

        # Ídem con la etiqueta
        label = float(res["labels"][local_id])

        # Convierto a tensores de PyTorch y devuelvo la pareja (x, y):
        #   - torch.from_numpy(window): envuelve el array numpy SIN copiar la memoria
        #   - torch.tensor(label, dtype=torch.float32): un tensor ESCALAR
        # El DataLoader junta muchas de estas parejas en un batch y lo devuelve
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.float32)

    # Labels e índices (para el sampler balanceado y el pos_weight)
    def positive_indices(self) -> np.ndarray:
        """Índices globales de las ventanas de crisis."""
        self._ensure_index()
        return np.flatnonzero(self._labels == 1)

    def negative_indices(self) -> np.ndarray:
        """Índices globales de las ventanas sin crisis."""
        self._ensure_index()
        return np.flatnonzero(self._labels == 0)

    @property
    def n_positive(self) -> int:
        return int(self.positive_indices().shape[0])


# la clase que me arma un epoch de entrenamiento. Cada vez q lo recorro (con __iter__), decide qué ventanas entran,
#las busca, las procesa, las amontona y las entrega. NO AMRA LOS BATCHES. ESO LO ARMA EL DATALOADER. Entonces si por época yo laburo con 100 archivos, devuelve los 100, aunque después mi batch sea de a 5
class BalancedEpochDataset(IterableDataset):
    """
    Época de entrenamiento balanceado, NO LAZY (guardo el array numpy con las ventanas elegidas), con shuffle global sobre ese array.

    El shuffle global es una masa (batches representativos, mezclados y balanceados). 
    Pero `process_edf` procesa el archivo entero y la RAM tiene cache de 1 archivo. 
    Entonces, si el DataLoader pide ventanas en orden aleatorio, muy probablemente tengo q reprocesar el
    archivo entero para servir una ventana. medio inviable entonces CADA ÉPOCA hago esto, en este orden:

      1) ELIJO el muestreo de la época con UNDERSAMPLING = en lugar de usar TODAS las negativas, uso solo una parte, elegida al azar, para compensar el desbalance (+ todas las positivas obvio)
        Es ALEATORIO con semilla fija, y una misma ventana negativa no aparece 2 veces. mi NEG_POS_RATIO es un hiperparámetro con el cual tengo q jugar e ir probando
      2) AGRUPO esas ventanas elegidas por ARCHIVO. Solo sirve para saber cuántas ventanas locales saco a cada archivo
      3) PROCESO cada archivo UNA SOLA VEZ (process_edf, ~1 s) y me quedo SOLO con las ventanas que salieron elegidas en el paso 1
      4) JUNTO todo en un array gigante y lo SHUFFLEO COMPLETO en RAM (orden q no respeta los archivos)
      5) ENTREGO las ventanas de a una y el DataLoader arma batches

    el modelo ve batches igual de mezclados y balanceados quecon el shuffle global clásico
    Hereda de `IterableDataset` porque la época es una SECUENCIA que se arma y yo la puedo recorrer una sola vez en ese orden
    """

    def __init__(self, dataset: WindowDataset, neg_pos_ratio: int | float = NEG_POS_RATIO,
                 seed: int = SEED):
        self.dataset = dataset
        self.neg_pos_ratio = neg_pos_ratio
        # Semilla reproducible: se consume UNA vez por época
        self.rng = np.random.default_rng(seed)

    def _pick_indices(self) -> np.ndarray:
        """PASO 1: elige los índices globales de la época. Todas las ventanas de crisis + NEG_POS_RATIO x crisis de ventanas no-crisis
        """
        self.dataset._ensure_index()  # por si el índice global aún no se armó
        pos = self.dataset.positive_indices()
        neg = self.dataset.negative_indices()
        n_pos = int(pos.shape[0])
        if n_pos == 0:
            return np.arange(len(self.dataset))
        n_neg = min(int(n_pos * self.neg_pos_ratio), len(neg))
        chosen_neg = self.rng.choice(neg, size=n_neg, replace=False)
        return np.concatenate([pos, chosen_neg])

    def _group_by_file(self, global_indices: np.ndarray) -> dict[Path, np.ndarray]:
        """PASO 2: agrupa los índices globales elegidos por archivo.

        Devuelve un diccionario con path_del_archivo: array con las ventanas LOCALES pedidas en ese archivo

        Uso los atributos del índice que ya armó el dataset (_file_ids, _local_ids, _valid_files).
        """
        file_ids = self.dataset._file_ids[global_indices]
        local_ids = self.dataset._local_ids[global_indices]
        groups: dict[Path, np.ndarray] = {}
        for file_id in np.unique(file_ids):
            path = self.dataset._valid_files[int(file_id)]
            mask = file_ids == file_id
            groups[path] = local_ids[mask]
        return groups

    @property
    def size_for_epoch(self) -> int:
        """Cantidad de ventanas de la época. Para ponerlo en los logs antes de iterar nom+as
        """
        self.dataset._ensure_index()
        n_pos = self.dataset.n_positive
        if n_pos == 0:
            return len(self.dataset)
        return n_pos + min(int(n_pos * self.neg_pos_ratio), len(self.dataset.negative_indices()))

    def __iter__(self):
        #muestreo de la época (crisis + negativas al azar)
        indices = self._pick_indices()

        # agrupar esas ventanas por archivo. me devuelve
        #{archivo 1: ([ventana10, ventana 25, ventana 50],
        # archivo 2: ([ventana 13]),
        # archivo 3: ([ventana 200, ventana 49])}
        groups = self._group_by_file(indices)

        # PASO 3: procesar cada archivo UNA SOLA VEZ y quedarse solo con las ventanas elegidas de cada uno. `_get_processed` es el caché
        # de 1 archivo del dataset: como recorro los archivos de a uno en orden, cada archivo se procesa exactamente una vez.
        win_blocks: list[np.ndarray] = []
        lbl_blocks: list[np.ndarray] = []
        for path, local_ids in groups.items(): #estp me da pares. k es len(local_ids)
            res = self.dataset._get_processed(path)          # proceso ese archivo puntual, completo
            win_blocks.append(res["windows"][local_ids])     #res[windows] son todas las ventanas del archivo. y con [local_ids] elijo las filas cuyos índices están en local ids. Me queda: (k, 16, 1310)
            lbl_blocks.append(res["labels"][local_ids])      # ídem q con windows! etiquetas de esas k

        # PASO 4: juntar todo y shufflear COMPLETO en RAM.
        if win_blocks:
            windows = np.concatenate(win_blocks, axis=0)     # concateno todos los win blocks, tabla única donde cada fila es una ventana, (N_epoca, 16, 1310)
            labels = np.concatenate(lbl_blocks, axis=0)      # (N_epoca,)
        else:
            windows = np.empty((0, N_CHANNELS, WIN_SAMPLES), dtype=np.float32)
            labels = np.empty(0, dtype=np.int8)
        # permutation(N) me devuelve los números del 0..N-1 pero desordenados; con ese orden reordeno ventanas y etiquetas JUNTAS (mismo orden para ambas así no quedan desalineadas).
        order = self.rng.permutation(len(windows))
        windows = windows[order]
        labels = labels[order]

        # PASO 5: entregar de a una. El DataLoader arma los batches
        for window, label in zip(windows, labels):
            #el for ... yield convierte este __iter__ en un generador. Cada vuelta genera (ventana, etiqueta) como tensores, se lo da a quien lo pidió y
            #queda en pausa hasta q le pidan el próximo. Es el dataloader el q va pidiendo
            yield torch.from_numpy(window), torch.tensor(label, dtype=torch.float32)

#mi dataset está desbalanceado pq tengo muchas más sin crisis q con crisis.
#eso afecta en mi función de pérdida (yo uso BCEWithLogits, (Binary Cross Entropy, la estándar para clasificación binaria), la cual mide cuánto se equivoca la red y pondera esos
#errores (los de todas las ventanas del batch) para calcular cuánto corregir los pesos. Pero mi desbalance hace que la sensibilidad baje,
#Porque el error de decir q una crisis sea positiva y q no lo sea "termina pesando más" que el caso contrario
#entonces acá lo que hago es hacer que justamente el error por decir q una ventana no tiene crisis cuando si la tiene pese mucho más (pos_weight) veces más, así la red no la ignora
#el peso se calcula a partir del ratio que se está usando, para q tampoco me sobredetecte crisis
# x ej ratio 3 -> pos_weight = 3  (cada crisis pesa 3x una normal en el loss)
def compute_positive_weight(dataset: WindowDataset, neg_pos_ratio: int | float = NEG_POS_RATIO) -> float:
    # Si el split no tiene ninguna crisis, el peso es irrelevante
    if dataset.n_positive == 0:
        return 1.0
    # Peso coherente con el undersampling: coincide con el ratio efectivo.
    return float(neg_pos_ratio)


# Construcción de Dataset + DataLoader por split
#solo la llamo desde train (Bah, desde build splits, pero a esa la llamo desde train)
def make_dataloader(dataset: WindowDataset, *, balanced: bool, batch_size: int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS, neg_pos_ratio: int | float = NEG_POS_RATIO,
                    seed: int = SEED) -> DataLoader:
    """DataLoader de un Dataset. `balanced=True` usa BalancedEpochDataset; `balanced=False` recorre TODAS las ventanas en orden (val/test, distribución natural)."""
    if balanced:
        # La época vive en UN proceso: si hubiera varios workers, cada uno armaría SU propia época completa y el modelo vería datos
        # duplicados. Por eso fuerzo 0 acá
        if num_workers != 0:
            print(f"[WARN] data.py: loader balanceado exige num_workers=0 "
                  f"se ignora num_workers={num_workers}.")
        epoch_ds = BalancedEpochDataset(dataset, neg_pos_ratio=neg_pos_ratio, seed=seed)
        return DataLoader(epoch_ds, batch_size=batch_size, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)

#solo la llamo desde train
def build_splits_dataloaders(data_dir: Path | str, split: dict, scaler_stats=None, *,  batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, neg_pos_ratio: int | float = NEG_POS_RATIO, seed: int = SEED, limit_files: int | None = None) -> dict:
    """
    Datasets y DataLoaders de train/val/test desde `split.json`.
    Devuelve:
        {"train": {"dataset":..., "dataloader":...},
         "val":   {"dataset":..., "dataloader":...},
         "test":  {"dataset":..., "dataloader":...}}

    - train  -> BalancedEpochDataset (balanceado + shuffle global)
    - val/test -> distribución natural (completa, sin tocar)
    """
    annotations = load_annotations(data_dir)
    by_name = annotations_by_name(annotations)
    out: dict[str, dict] = {}
    for name in ("train", "val", "test"):
        files = files_for_patients(data_dir, split.get(name, []))
        if limit_files is not None:
            files = files[:limit_files]
        dataset = WindowDataset(files, by_name, scaler_stats=scaler_stats)
        dataloader = make_dataloader(dataset, balanced=(name == "train"),
                                     batch_size=batch_size, num_workers=num_workers,
                                     neg_pos_ratio=neg_pos_ratio, seed=seed)
        out[name] = {"dataset": dataset, "dataloader": dataloader}
    return out

#diagnóstico para ver cómo salieron los valores ya escalados
#valores ADIMENSIONALES, calculados sobre las ventanas ya escaladas.
def _print_batch(name: str, windows, labels) -> None:
    mean = windows.float().mean().item() #media del canal entero
    std = windows.float().std().item()
    print(f"   {name:>6}: batch{batch_shape(windows)}  positivas={int(labels.sum().item())}/{labels.shape[0]}  "
          f"media={mean: .3f}  std={std: .3f}")


def batch_shape(windows) -> tuple[int, ...]:
    return tuple(windows.shape)


def smoke_cli(args) -> None:
    data_dir = Path(args.data_dir)
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))

    scaler_stats = load_scaler_stats(SCALER_STATS_FILE)
    if scaler_stats is None:
        print("[WARN] No existe scaler_stats.npz: las ventanas salen SIN estandarizar.")
    else:
        print(f"[ok] scaler_stats: '{scaler_stats['scaler']}', {scaler_stats['n_windows']} "
              f"ventanas de train (archivos: {scaler_stats['n_files']})")

    patients = split.get("train", [])
    files = files_for_patients(data_dir, patients)
    if args.limit_files:
        files = files[: args.limit_files]
    print(f"\nPacientes de train: {patients}")
    print(f"Archivos a usar (limit={args.limit_files}): " f"{[f.name for f in files]}")

    dataset = WindowDataset(files, annotations_by_name(load_annotations(data_dir)), scaler_stats=scaler_stats)
    print(f"\nÍndice global (solo cabeceras): {len(dataset)} ventanas de " f"{len(dataset._valid_files)} archivos válidos")
    print(f"   positivas={dataset.n_positive}  negativas={len(dataset) - dataset.n_positive}")

    pos_weight = compute_positive_weight(dataset)
    print(f"   pos_weight: {pos_weight:.1f}")

    epoch = BalancedEpochDataset(dataset, seed=args.seed)
    #envuelvo el epochdataset en un dataloader q arma batches de a batch size (default 8)
    loader = DataLoader(epoch, batch_size=args.batch_size)
    print(f"\nÉpoca de train balanceada + shuffle global: "
          f"{epoch.size_for_epoch} ventanas (NEG_POS_RATIO={NEG_POS_RATIO})")

    print(f"\nPrimeros {args.max_batches} batches:")
    #recorro el dataloader. Enumerate me da el número de batch y el contenido (window, label)
    for i, (windows, labels) in enumerate(loader):
        #window es un tensor con forma (batch, 16, 1310), label forma (batch)
        if i >= args.max_batches: #No recorro la época completa, solamente los primeros 5
            break
        _print_batch(f"batch{i}", windows, labels)
        if i == 0:
            first = windows[0]
            print(f"   shape ventana: {tuple(first.shape)}  dtype: {first.dtype}")
            print(f"   canal FP1-F7: media={first[0].mean(): .3f}  std={first[0].std(): .3f}")

    print("\nSmoke-test OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset/DataLoader")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--smoke", action="store_true", help="Smoke-test con pocos archivos.")
    parser.add_argument("--limit-files", type=int, default=3, help="Máx. de EDFs en el smoke.")
    parser.add_argument("--max-batches", type=int, default=5, help="Batches a imprimir.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.smoke:
        smoke_cli(args) #test para ver q no se rompa nada
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()