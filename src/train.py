"""
Entrenamiento de la CNN-1D (en mi google colab pro).

Junta `data.py` con `model.py` y entrena la red para clasificar cada ventana de EEG.

    python -m src.train                    # corrida completa (CPU o GPU según haya)
    python -m src.train --limit-files 6    # smoke de entrenamiento (pocos EDFs)

  1. Carga split.json y scaler_stats.npz
  2. Arma los DataLoaders: train balanceado y shuffleado (BalancedEpochDataset) y val natural (sin tocar la distribución).
  3. Instancia la CNN, el optimizador AdamW y el loss BCEWithLogits con pos_weight.
  4. Corre las épocas: por cada batch -> ventanas -> red -> loss -> backward -> paso del optimizador. Al final de cada época evalúa en validación.
  5. Early stopping: si el loss de val no mejora en PATIENCE épocas, corta.
  6. Guarda el mejor modelo en models/best.pt (pesos + config + stats de escalado para poder inferir después sin re-entrenar).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    CONV_CHANNELS,
    CONV_KERNELS,
    DATASET_DIR,
    DROPOUT,
    EPOCHS,
    FC_UNITS,
    FS,
    LEARNING_RATE,
    MODELS_DIR,
    N_CHANNELS,
    NEG_POS_RATIO,
    NUM_WORKERS,
    PATIENCE,
    SCALER_STATS_FILE,
    SEED,
    SPLIT_FILE,
    STRIDE_SAMPLES,
    THRESHOLD,
    WEIGHT_DECAY,
    WIN_SECONDS,
)
from src.data import build_splits_dataloaders, compute_positive_weight
from src.model import SeizureCNN
from src.preprocessing import load_scaler_stats



def set_seed(seed: int) -> None:
    """
    Fija TODO el azar del entrenamiento para que sea reproducible (dos corridas con el mismo SEED dan los mismos valores)
      - random / numpy: shuffle de índices y submuestreos.
      - torch.manual_seed: inicialización de pesos y orden de batches.
      - cudnn determinístico: en GPU, que las convoluciones den siempre igual.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True fuerza algoritmos reproducibles (algo más lentos);
        # benchmark=False evita que cuDNN elija kernels "a la carrera".
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# una época de entrenamiento y la evaluación en validación
def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> float:
    """
    Por batch:
      1. mueve ventanas y labels al device
      2. `model(x)` = forward -> logits (B, 1);
      3. `criterion` (BCEWithLogits + pos_weight) compara logits vs. labels;
      4. `loss.backward()` = backpropagation: calcula el gradiente de cada peso;
      5. `optimizer.step()` = AdamW actualiza los pesos siguiendo esos gradientes.

    Devuelve el loss PROMEDIO de la época (suma pesada por tamaño de batch).
    """
    model.train()  # activa Dropout y las stats de batch del BatchNorm (modo entrenar)

    total_loss = 0.0
    n_seen = 0  # contador de ventanas vistas (para promediar bien el loss)

    for windows, labels in loader:
        windows = windows.to(device)   # (B, 16, 1310)
        labels = labels.to(device)     # (B,) etiquetas 0/1

        optimizer.zero_grad()          # resetea gradientes acumulados del batch anterior
       #Estas dos funciones escriben sus operaciones en el grafo de gradientes (se arma uno en cada batch), y los resultados intermedios se guardan en memoria
       #cada nodo del grafo guarda q operación lo creó (mul, add, sigmoid, log), de q tensores vino (Los inputs), los valores numéticos de los resultados (las activaciones q salieron, no los gradientes)
       #guardo los resultados parciales pq la derivada de una operación depende del valor de su input, por ende no puedo calcular el gradiente sin eso
        logits = model(windows)        # (B, 1) — el logit crudo, sin sigmoide
        loss = criterion(logits.squeeze(-1), labels)  # calculo el loss! BCEWithLogits espera logits (B,) y target (B,)
       #acá recorro ese grafo de gradientes en orden inverso, se calculan los gradientes, y libero el grafo.
        loss.backward()                # calculo todos los gradientes de todos los pesos usando backpropagation
        optimizer.step()               # ACÁ el optimizador aplica la fórmula de AdamW y actualiza los pesos posta

        # loss.item() es el loss PROMEDIO del batch; lo multiplico por el tamaño para acumular la suma total y después promediar sobre todas las ventanas.
        total_loss += loss.item() * windows.size(0)
        n_seen += windows.size(0)

    return total_loss / n_seen


@torch.no_grad()  # no construye grafo de gradientes para ahorra memoria, total no actualizo los pesos
def evaluate(model: nn.Module, loader, criterion, device: str) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evalúa el modelo sobre un DataLoader (validación o test)
    Devuelve:
      - val_loss (promedio, con el criterion SIN pos_weight);
      - probs: todas las probabilidades
      - labels: todas las etiquetas reales
    """
    model.eval()  # apaga Dropout y usa running_mean/var del BatchNorm (modo evaluar)

    total_loss = 0.0
    n_seen = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
#tensor de forma (64, 16, 1310) para windows y (64,) par alabels
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)
        logits = model(windows)                  # (B, 1)
        loss = criterion(logits.squeeze(-1), labels)  # loss sin pos_weight (natural)
        total_loss += loss.item() * windows.size(0)
        n_seen += windows.size(0)
        
        # le saco la últma dimensión a logits, pasa de (64, 1) a (64). Convierto cada uno a prob con la sigmoide, lo muevo a la CPU, y lo vot acynulando
        all_probs.append(torch.sigmoid(logits.squeeze(-1)).cpu())
        all_labels.append(labels.cpu()) #ídem

    #concatena todos los tensores de cada batch y me deja un solo tensor. 
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    #total_loss/n_seen = VAL LOSS
    return total_loss / n_seen, probs, labels


def binary_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    """
    Métricas clínicas de clasificación binaria sobre TODAS las ventanas de un
    split (val o test), usando el umbral de decisión.

    Terminología:
      - Sensibilidad (Recall) = qué tan bien detecta crisis.
      - Especificidad          = qué tan bien descarta normal.
      - FPR (tasa)             = 1 - Especificidad (fracción de negativas marcadas como crisis).
      - FP/hora                = falsos positivos por hora de grabación (conteo / horas).
      - Accuracy               = (actual_positive + actual_negative) / total.
    """
    pred = (probs >= threshold).float()  # aplica el umbral -> rótulo binario
    # Matriz de confusión en conteos:
    actual_positive = int(((pred == 1) & (labels == 1)).sum())  # TP: crisis bien detectada
    false_positive = int(((pred == 1) & (labels == 0)).sum())   # FP: alarma falsa
    actual_negative = int(((pred == 0) & (labels == 0)).sum())  # TN: normal bien descartada
    false_negative = int(((pred == 0) & (labels == 1)).sum())   # FN: crisis no detectada

    # Horas de grabación cubiertas por las ventanas del split
    # las ventanas solapan 50%, así que cada ventana NUEVA aporta solo STRIDE_SAMPLES muestras de señal (no WIN_SAMPLES). Si usara
    # n_windows * WIN_SAMPLES estaría contando el doble de tiempo y el fp/h saldría la mitad del valor real.
    n_windows = int(labels.numel())
    # span_seconds = tiempo entre el inicio de la 1er ventana y el fin de la última.
    span_seconds = (n_windows - 1) * (STRIDE_SAMPLES / FS) + WIN_SECONDS if n_windows > 0 else 0.0
    total_hours = span_seconds / 3600.0

    total = actual_positive + false_positive + actual_negative + false_negative
    return {
        "sensibility": actual_positive / (actual_positive + false_negative) if (actual_positive + false_negative) > 0 else 0.0,
        "specificity": actual_negative / (actual_negative + false_positive) if (actual_negative + false_positive) > 0 else 0.0,
        # TASA de falsos positivos: fracción de las negativas marcadas como crisis. 1-specificity
        "false_positive_rate": false_positive / (false_positive + actual_negative) if (false_positive + actual_negative) > 0 else 0.0,
        # Falsos positivos POR HORA: conteo real normalizado por tiempo de grabación. métrica clínica estándar en detección de crisis
        "false_positive_per_hour": false_positive / total_hours if total_hours > 0 else 0.0,
        "accuracy": (actual_positive + actual_negative) / total if total > 0 else 0.0,
        # Conteos crudos
        "actual_positive": actual_positive, "false_positive": false_positive, "actual_negative": actual_negative, "false_negative": false_negative,
    }


def save_checkpoint(path: Path, model: nn.Module, scaler_stats: dict, *,
                    epoch: int, best_val_loss: float, pos_weight: float, seed: int) -> None:
    """
    Guarda lo necesario para re-usar el modelo sin re-entrenar:

      - model_state_dict: los valores de TODOS los pesos (la "memoria" aprendida);
      - config del modelo: listas de canales/kernels, fc_units, dropout, etc. (para reconstruir la arquitectura idéntica al cargar);
      - scaler_stats: mediana/IQR por canal del RobustScaler (para que inference escale las ventanas nuevas EXACTAMENTE igual que en train);
      - metadata: época, best_val_loss, pos_weight, seed (auditoría).

    Se guarda en disco el momento exacto en que el loss de validación fue mínimo
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": N_CHANNELS,
            "conv_channels": list(CONV_CHANNELS),
            "conv_kernels": list(CONV_KERNELS),
            "fc_units": FC_UNITS,
            "dropout": DROPOUT,
        },
        "scaler_stats": {
            "channels": scaler_stats["channels"],
            "median": scaler_stats["median"],
            "iqr": scaler_stats["iqr"],
        },
        "threshold": THRESHOLD,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "pos_weight": pos_weight,
        "seed": seed,
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    # Los flags permiten ajustar la corrida sin tocar config.py (para Colab, qpaso --data-dir y --split-file con las rutas de Drive)
    parser = argparse.ArgumentParser(description="Entrenamiento CNN-1D")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--scaler-stats", type=str, default=str(SCALER_STATS_FILE))
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "best.pt"),  help="Ruta del checkpoint a guardar.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--neg-pos-ratio", type=float, default=NEG_POS_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--limit-files", type=int, default=None, help="Limitar a N EDFs por split (smoke-test rápido).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}  (semilla={args.seed})")

    # Carga del split y de las stats del escalador
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    scaler_stats = load_scaler_stats(Path(args.scaler_stats))
    if scaler_stats is None:
        sys.exit("No existe scaler_stats.npz. Corré primero:\n""    python -m src.preprocessing --compute-stats\n")

    loaders = build_splits_dataloaders(args.data_dir, split, scaler_stats, batch_size=args.batch_size,
        num_workers=args.num_workers, neg_pos_ratio=args.neg_pos_ratio, seed=args.seed, limit_files=args.limit_files,)
    #solo uso esto para tener info del dataset e imprimirlo. Sino uso el dataloader
    train_ds = loaders["train"]["dataset"]
    train_loader = loaders["train"]["dataloader"]
    val_loader = loaders["val"]["dataloader"]

    pos_weight = compute_positive_weight(train_ds, neg_pos_ratio=args.neg_pos_ratio)

    print(f"\nConfiguración de entrenamiento")
    print(f"Train: {len(train_ds)} ventanas ({train_ds.n_positive} positivas, "
          f"{len(train_ds) - train_ds.n_positive} negativas)")
    print(f"pos_weight: {pos_weight:.1f}  |  NEG_POS_RATIO: {args.neg_pos_ratio}")
    print(f"batch_size={args.batch_size}  lr={args.lr}  weight_decay={args.weight_decay}  "
          f"epochs={args.epochs}  patience={args.patience}")

    model = SeizureCNN().to(device)

    # AdamW es Adam + weight decay DESACOPLADO (hat dos formas de meterlo: acoplado, agreganfo el castigo al gradiente, o el desacoplado)
    # El adam solito usa momentum y tasa adaptativa. El weight decay sirve para evitar el overfitting. Penaliza pesos
    # grandes (en vada paso achico cada peso hacia 0) haciendolo DIRECTAMENTE en el peso, sin mezclarse con el momento adaptativo (regularización L2, en donde L2= cuadrado de los pesos. Entoncees
    # mi loss queda L_total= L + (weight_decay/2) * sumatoria de(weights^2))
    #entonces, acá creo el optimizador AdamW, le paso la lista de todos los pesos, el learning rate inicial, y el weight decay de cuánto achico los pesos.
    #NO HACE NADA TODAVÍA, solo lo creo!
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   #La BCE mide el error de mi predicción con la fórmula L = −[ y·log(p) + (1−y)·log(1−p) ], con p la probabilidad de crisis q predije e y=0 o y=1 según si hay o no
   #la probabilidad p es básicamente la sigmoide de mis logits (los logits son la salida real de mi cnn, yo con la sigmoide q es 1/(1+e^(-logit)) lo llevo a p entre 0 y 1)
   #el BCE with logits me hace la transformación a probabilidad internamente así no la hago yop
   #toma el promedio de los errores individuales de cada ventana del batch
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    # Loss de VALIDACIÓN. Independientemente si en la función de pérdida usé o no el pos_weight, acá NO lo uso. Necesito saber si la red generaliza, de forma estable
    val_criterion = nn.BCEWithLogitsLoss()

    # Loop de entrenamiento con early stopping si no mejora en X cantidad de épocas
    best_val_loss = float("inf")  # cualquier pérdida inicial mejora
    best_epoch = 0
    patience_left = args.patience
    start = time.time()
    history: list[dict] = []  # métricas por época (para las curvas de loss/accuracy en la tesis)

    print(f"\n=== Entrenamiento ({args.epochs} épocas máx.) ===")
    print(f"{'Ep':>3} | {'train_loss':>12} | {'val_loss':>12} | {'sens':>10} | {'spec':>10} | "
          f"{'fpr':>10} | {'fp/h':>10} | {'acc':>10} | {'tiempo':>8}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        #TRAIN LOSS SE CALCULA SOBRE LOS DATOS DE TRAIN, Q ESTÁN BALANCEADOS. SE MIDE MIENTRAS EL MODELO SE ACTUALIZA
        #VAL LOSS SE CALCULA SOBRE LOS PACIENTES DE VALIDACIÓN (NO LOS DE TEST TODAVÍA), ENTONCES NO TIENEN UNDERSAMPLING, NI POS WEIGHT,
        #NI DROPOUT. O SEA VAL LOSS ME DICE Q TAN BIEN GENERALIZÓ EN PROMEDIO FRENTE A DATOS CON LOS Q NO ENTRENÉ
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss, probs, labels = evaluate(model, val_loader, val_criterion, device)
        m = binary_metrics(probs, labels, THRESHOLD)

        elapsed_time_for_epoch = time.time() - t_epoch
        print(f"{epoch:>3} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{m['sensibility']:>10.4f} | {m['specificity']:>10.4f} | {m['false_positive_rate']:>10.4f} | "
              f"{m['false_positive_per_hour']:>10.4f} | {m['accuracy']:>10.4f} | {elapsed_time_for_epoch:>7.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sensibility": m["sensibility"],
            "specificity": m["specificity"],
            "false_positive_rate": m["false_positive_rate"],
            "false_positive_per_hour": m["false_positive_per_hour"],
            "accuracy": m["accuracy"],
            "time_s": elapsed_time_for_epoch,
        })

        #si val_loss mejoró, guardo el checkpoint y reseteo paciencia
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(
                Path(args.out), model, scaler_stats,
                epoch=epoch, best_val_loss=best_val_loss,
                pos_weight=pos_weight, seed=args.seed,
            )
            print(f"    -> mejor val_loss={best_val_loss:.6f}, checkpoint guardado en {args.out}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping: val_loss no mejoró en {args.patience} épocas. Cortando.")
                break

    total_time = time.time() - start
    print(f"\n=== Fin del entrenamiento ===")
    print(f"Mejor val_loss: {best_val_loss:.6f}")
    print(f"Tiempo total: {total_time / 60:.1f} min")
    print(f"Checkpoint: {args.out}")

    # Guarda el historial por época (curvas loss/accuracy para la tesis). Se persiste
    # junto al checkpoint como <nombre>.history.json para que el notebook lo pueda leer sin re-entrenar.
    history_path = Path(args.out).with_name(Path(args.out).stem + ".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "threshold": THRESHOLD,
        "pos_weight": pos_weight,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"Historial de entrenamiento: {history_path}")


if __name__ == "__main__":
    main()
