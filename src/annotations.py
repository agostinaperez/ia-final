"""
Parser de las anotaciones clínicas de CHB-MIT.

CHB-MIT viene con un archivo por paciente llamado `chbXX-summary.txt` que lista, para cada archivo `.edf`, los intervalos de crisis ("Seizure Start Time" /
"Seizure End Time") en segundos. Este módulo lo convierte en estructuras Python listas para usar (por paciente -> por archivo -> lista de intervalos).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.config import DATASET_DIR

# Expresiones regulares para detectar las líneas relevantes del summary.
# - "File Name: chb01_03.edf"      -> empieza un nuevo archivo.
# - "Seizure Start Time: 2996 seconds"  (formato de chb01–chb05)
# - "Seizure 1 Start Time: 1665 seconds" (formato numerado, chb06 en adelante)
# OJO: hay DOS formatos según el paciente. El "(?:\d+\s+)?" de los regex
# acepta el "1 ", "2 ", ... opcional para que matchee los dos.
_RE_FILE = re.compile(r"^File Name:\s+(.+)")  # captura el nombre del EDF
_RE_START = re.compile(r"^Seizure\s+(?:\d+\s+)?Start Time:\s+(\d+)")
_RE_END = re.compile(r"^Seizure\s+(?:\d+\s+)?End Time:\s+(\d+)")


def discover_patients(data_dir: Path | str = DATASET_DIR) -> list[str]:
    data_dir = Path(data_dir)

    # Lista de pacientes: carpetas que empiezan con "chb" y contienen EDFs.
    patients = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.lower().startswith("chb"):
            continue
        # Un paciente cuenta si efectivamente tiene archivos EDF dentro.
        if any(child.glob("*.edf")):
            patients.append(child.name)

    if not patients:
        raise FileNotFoundError(f"No se encontraron carpetas de pacientes 'chbNN' en: {data_dir}")

    return patients


def summary_path_for(patient: str, data_dir: Path | str = DATASET_DIR) -> Path:
    #Ruta del archivo de resumen de un paciente: `<carpeta>/<paciente>-summary.txt`.
    return Path(data_dir) / patient / f"{patient}-summary.txt"


def parse_summary(summary_path: Path | str) -> dict[str, list[tuple[int, int]]]:
    """
    Parsea un archivo summary y devuelve:

        {"chb01_03.edf": [(2996, 3036)], "chb01_04.edf": [(1467, 1494)], ...}

    - Cada clave es el nombre de un archivo EDF.
    - Cada valor es una lista de tuplas (inicio_seg, fin_seg) de crisis.
    - Se devuelve una lista vacía para los archivos sin crisis (así el split sabe que existen aunque no tengan eventos).
    """

    current_file: str | None = None
    # acá guardo el valor de inicio de la crisis (me falta guardarle el end time)
    pending_start: int | None = None

    # Mapa nombre->lista de intervalos
    seizures: dict[str, list[tuple[int, int]]] = {}

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()  # limpiar saltos de línea

            # la línea "File Name: ..." indica q cambié de archivo
            m = _RE_FILE.match(line)
            if m:
                current_file = m.group(1).strip()
                # Inicializo la lista de crisis para este arvchivo (si no tiene crisis, queda vacío y listo)
                seizures.setdefault(current_file, [])
                # reinicio esto para q no queden valores cruzados
                pending_start = None
                continue

            if current_file is None:
                continue

            m = _RE_START.match(line)
            if m:
                pending_start = int(m.group(1)) #guardo el inicio de la crisis
                continue

            m = _RE_END.match(line)
            if m and pending_start is not None:
                end = int(m.group(1))
                # no puede terminar antes de empezar.
                if end >= pending_start:
                    seizures[current_file].append((pending_start, end)) #formo el invervalo de segundos de inicio y de fin de la crisis
                pending_start = None #limpio el valor
    return {k: v for k, v in seizures.items() if k} #devuelvo solamente los valores con nombre de archivo no vacío x si quedó alguna línea rari


def load_annotations(data_dir: Path | str = DATASET_DIR) -> dict[str, dict]:
    """
    Carga las anotaciones de todos los pacientes del dataset.
    Devuelve: {"chb01": {"chb01_03.edf": [(2996, 3036), ...], ...}, ...}
    """
    data_dir = Path(data_dir)
    annotations: dict[str, dict] = {}

    for patient in discover_patients(data_dir):
        summary = summary_path_for(patient, data_dir)
        if summary.exists():
            annotations[patient] = parse_summary(summary)
        else:
            print(f"[WARN] {patient}: no tiene {summary.name}; se omite.")

    return annotations


def patient_seizure_stats(annotations: dict[str, dict],) -> list[dict]:
    """
    Resumen de severidad por paciente, útil para el split proporcional.

    Para cada paciente calcula:
      - patient:     código del paciente
      - n_files:     cantidad de archivos EDF anotados (cada uno ~1 hora)
      - n_seizures:  cantidad total de eventos de crisis
      - seizure_seconds: duración total de crisis en segundos

    Se usa n_files como proxy del volumen de registro de cada paciente, y
    seizure_seconds como proxy de la "cantidad de positivos" que aporta.
    """
    stats = []

    for patient, files in annotations.items():
        # n_files: cuántos EDF menciona el summary para este paciente.
        n_files = len(files)

        # Concentramos todos los intervalos de crisis del paciente.
        all_intervals = []
        for ivs in files.values():      # 1. por cada lista de intervalos
            for iv in ivs:              # 2. por cada intervalo dentro de esa lista
                all_intervals.append(iv)

        # Cantidad de eventos de crisis.
        n_seizures = len(all_intervals)

        # Suma de duraciones de todas sus crisis (en segundos).
        seizure_seconds = sum(end - start for start, end in all_intervals)

        stats.append(
            {
                "patient": patient,
                "n_files": n_files,
                "n_seizures": n_seizures,
                "seizure_seconds": seizure_seconds,
            }
        )

    return stats


if __name__ == "__main__":
    # Imprime, por paciente, la cantidad de crisis y sus segundos totales.
    import json

    annotations = load_annotations()
    stats = patient_seizure_stats(annotations)
    #imprime corte
        #paciente  archivos  crisis  segundos
        #chb01         4        8      3589
        #chb10        12        1       340
    
    print(f"{'paciente':<8} {'archivos':>8} {'crisis':>6} {'segundos':>10}")
    for s in sorted(stats, key=lambda x: x["patient"]):
        print(
            f"{s['patient']:<8} {s['n_files']:>8} "
            f"{s['n_seizures']:>6} {s['seizure_seconds']:>10}"
        )
    print("\nResumen:", json.dumps(annotations.get("chb01"), indent=2))