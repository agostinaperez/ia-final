"""
Split inter-paciente 70/30 proporcional

Asigna pacientes COMPLETOS a Train/Val/Test para la generalización inter-paciente y q no haya fuga de datos.

La asignación se hace por orden de "severidad" (segundos de crisis) para que train y test tengan una mezcla parecida de pacientes "graves" (con muchas crisis).
Se usa una asignación codiciosa que mantiene, tanto en volumen de registro (n_files) como en duración de crisis (seizure_seconds), la fracción de test
cerca de TEST_RATIO (30%).

Salida: escribe data/processed/split.json con los conjuntos y el detalle.

Uso:
    python -m src.split
    python -m src.split --data-dir "C:/ruta/al/dataset" --out "ruta/split.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.annotations import load_annotations, patient_seizure_stats
from src.config import (
    DATASET_DIR,
    N_VAL_PATIENTS,
    PROCESSED_DIR,
    SPLIT_W_ZSEC,
    TEST_RATIO,
)


def split_dataset(
    stats: list[dict],
    test_ratio: float = TEST_RATIO,
    w_zsec: float = SPLIT_W_ZSEC,
) -> tuple[list[str], list[str]]:
    """
    Ordena por severidad DESCENDENTE (más crisis primero)
    Recorre en ese orden y decide si va a train o test, según cuál opción deja la fracción acumulada de "volumen" GLOBAL más cercana a (1 - test_ratio).
    El "volumen" se mide con dos métricas combinadas:
        - fracción de archivos (peso 1.0)
        - fracción de segundos de crisis (peso w_zsec, por defecto 2.0)

    w_zsec > 1 hace que la balanza mire más los segundos de crisis, que son
    los que definen cuántas ventanas POSITIVAS quedan en test.

    Devuelve (train, test) con los códigos de paciente.
    """
    # 1) Orden de "severidad": pacientes con más crisis primero, así los
    #    "graves" no se van todos juntos para el mismo lado al final.
    ordered = sorted(stats, key=lambda s: (s["seizure_seconds"], s["n_files"]), reverse=True)

    # Totales sobre todos los pacientes (el "denominador" para medir fracciones).
    # El "or 1" es un seguro por si todo diera cero (evita dividir por 0).
    total_files = sum(s["n_files"] for s in ordered) or 1
    total_zsec = sum(s["seizure_seconds"] for s in ordered) or 1

    # La meta: train debe quedarse con el 70% (lo que no va a test).
    target_train = 1.0 - test_ratio

    # Acumuladores: cuántos archivos / seg. de crisis ya repartimos a cada lado.
    train_files = test_files = 0
    train_zsec = test_zsec = 0.0

    train_patients: list[str] = []
    test_patients: list[str] = []

    # Reparto "codicioso": paciente por paciente, del más grave al más liviano.
    # Para cada paciente probamos LAS DOS OPCIONES (mandarlo a train o a test)
    # y elegimos la que deje el acumulado global MÁS cerquita del objetivo
    # (train = 70%). Así vamos "rellenando" el tren hasta llegar al 70%.
    for s in ordered:
        f = s["n_files"]  # cuánto "volumen" aporta: archivos grabados (~1h c/u)
        zsec = s["seizure_seconds"]  # cuántos segundos de crisis le aporta

        # OPCIÓN A: si este paciente va a TRAIN.
        # Train quedaría = lo que ya tenía acumulado + lo que aporta él.
        # Calculamos qué fracción del total global es ese acumulado.
        train_frac_files = (train_files + f) / total_files
        train_frac_zsec = (train_zsec + zsec) / total_zsec

        # OPCIÓN B: si este paciente va a TEST.
        # Train NO gana nada, su acumulado queda exactamente donde estaba.
        test_frac_files = train_files / total_files
        test_frac_zsec = train_zsec / total_zsec

        # Medimos "qué tan lejos" quedó cada opción del 70% buscado.
        # Sumamos la desviación en archivos (peso 1) + la desviación en seg. de
        # crisis (peso w_zsec). Valor chico = quedó cerquita del objetivo.
        dev_to_train = abs(train_frac_files - target_train) + w_zsec * abs(train_frac_zsec - target_train)
        dev_to_test = abs(test_frac_files - target_train) + w_zsec * abs(test_frac_zsec - target_train)

        # Elegimos la opción que menos se desvía (¿queda mejor en train o en test?).
        if dev_to_test < dev_to_train:
            test_patients.append(s["patient"])
            test_files += f
            test_zsec += zsec
        else:
            train_patients.append(s["patient"])
            train_files += f
            train_zsec += zsec

    return train_patients, test_patients


def pick_validation(train_patients: list[str], stats: list[dict], n_val: int = N_VAL_PATIENTS) -> list[str]:
    """
    Elige N pacientes de validación DENTRO del conjunto de train.

    se toman pacientes "del medio" en severidad para que la validación tenga una mezcla típica
    de eventos. Si no alcanza el número pedido, se toma lo que haya.
    """
    # Mapa paciente -> severidad para ordenar.
    sev = {s["patient"]: s["seizure_seconds"] for s in stats}

    # Ordenar los train por severidad ascendente (menos crisis -> más crisis).
    train_sorted = sorted(train_patients, key=lambda p: sev.get(p, 0))

    # Elegimos posiciones "centrales" repartidas uniformemente a lo largo del orden, evitando extremos y sin repetición
    assert len(train_sorted) > n_val, (
        "Se necesitan más pacientes de train que " f"{n_val} para reservar validación."
    )
    idx = [(k * len(train_sorted)) // (n_val + 1) for k in range(1, n_val + 1)]

    # Tomamos las n_val posiciones centrales disponibles.
    val = [train_sorted[i] for i in idx if i < len(train_sorted)]
    return val[:n_val]


def build_split(data_dir: Path | str = DATASET_DIR) -> dict:
    """
    Ejecuta el pipeline completo del split y devuelve el diccionario-resumen.

    El diccionario contiene:
      - data_dir, ratios
      - train / val / test: listas de pacientes
      - patients: detalle por paciente (split + stats)
      - resumen_volumen: fracciones logradas para control de proporcionalidad
    """
    data_dir = Path(data_dir)

    #Cargar anotaciones de todos los pacientes.
    annotations = load_annotations(data_dir)

    #Estadísticas de severidad por paciente.
    stats = patient_seizure_stats(annotations)
    if not stats:
        sys.exit("No hay pacientes con anotaciones")

    # Split determinista (w_zsec pesa más los segundos de crisis para equilibrar).
    train, test = split_dataset(stats, TEST_RATIO, SPLIT_W_ZSEC)

    #Reservo validación desde train
    val = pick_validation(train, stats, N_VAL_PATIENTS)

    #Quitar los de validación de la lista de train "efectiva"
    train_eff = [p for p in train if p not in set(val)]

    # Chequeo de integridad
    all_sets = train_eff + val + test
    assert len(all_sets) == len(set(all_sets)), "Paciente repetido entre splits."
    assert len(all_sets) == len(stats), "Se perdió algún paciente en el reparto."

    # Detalle por paciente para el reporte.
    split_of = {}
    for s in stats:
        p = s["patient"]
        if p in test:
            s["split"] = "test"
        elif p in val:
            s["split"] = "val"
        else:
            s["split"] = "train"
        split_of[p] = s

    # Resumen de volúmenes para verificar la proporcionalidad.
    def volume(pool):
        files = sum(s["n_files"] for s in stats if s["patient"] in pool)
        zsec = sum(s["seizure_seconds"] for s in stats if s["patient"] in pool)
        return {"files": files, "seizure_seconds": zsec}

    total_files = sum(s["n_files"] for s in stats)
    total_zsec = sum(s["seizure_seconds"] for s in stats)
    summary = {
        "data_dir": str(data_dir),
        "test_ratio_solicitado": TEST_RATIO,
        "n_train": len(train_eff),
        "n_val": len(val),
        "n_test": len(test),
        "train": train_eff,
        "val": val,
        "test": test,
        "patients": split_of,
        "volumen": {
            "train": volume(train_eff),
            "val": volume(val),
            "test": volume(test),
            "total": {"files": total_files, "seizure_seconds": total_zsec},
            "frac_test_files": round(
                volume(test)["files"] / total_files, 4
            ),
            "frac_test_seizure_seconds": round(
                volume(test)["seizure_seconds"] / total_zsec, 4
            ),
        },
    }
    return summary


def print_report(summary: dict) -> None:
    """Imprime una tabla legible con el resultado del split."""
    print("\n" + "=" * 60)
    print("SPLIT INTER-PACIENTE (70/30 proporcional)")
    print("=" * 60)
    header = f"{'paciente':<9} {'split':<6} {'archivos':>8} {'crisis':>6} {'segcrisis':>10}"
    print(header)
    print("-" * len(header))
    for p, s in sorted(summary["patients"].items()):
        print(
            f"{p:<9} {s['split']:<6} {s['n_files']:>8} "
            f"{s['n_seizures']:>6} {s['seizure_seconds']:>10}"
        )
    v = summary["volumen"]
    print("-" * len(header))
    print(f"TRAIN: {summary['n_train']} pacientes | "
          f"VAL: {summary['n_val']} | TEST: {summary['n_test']}")
    print(f"Frac test (archivos):         {v['frac_test_files']}")
    print(f"Frac test (segundos crisis):  {v['frac_test_seizure_seconds']}")
    print("=" * 60)


def main() -> None:
    """CLI: python -m src.split [--data-dir RUTA] [--out RUTA]"""
    parser = argparse.ArgumentParser(description="Split inter-paciente CHB-MIT.")
    # Permite correr en local y en Colab con paths distintos.
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--out", type=str, default=str(PROCESSED_DIR / "split.json"))
    args = parser.parse_args()

    # Generar el split
    summary = build_split(args.data_dir)

    # Mostrar y persistir
    print_report(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSplit guardado en: {out_path}")


if __name__ == "__main__":
    main()