# NILM Training (M-GRU Seq2Seq)

Dieses Repository enthält den Training- und Evaluationscode für ein NILM-Modell (Non-Intrusive Load Monitoring)
auf Basis eines GRU mit Seq2Seq-Output.
Ziel ist die Erzeugung gerätebezogener Zustandssequenzen („An“/„Aus“) aus 1-Hz-Summenleistung
(Smart-Meter), inklusive Artefakte und Auswertungen.

Der Code ist so strukturiert, dass er sowohl mit dem DEDDIAG-Datensatz (PostgreSQL)
als auch mit eigenen CSV-Messdaten betrieben werden kann.

## Inhalt und Funktionsumfang

- Datenloader
  - DEDDIAG (PostgreSQL): Laden von Mains + Gerätekanälen und zusammenfuehren auf gemeinsame Zeitachse
  - CSV: Laden Spaltenkonfigurationen inkl. Zeitschnitt
- Datenaufbereitung
  - Z-Normalisierung (Z-Score) auf Trainingssplit
  - Feature-Vektor pro Zeitschritt: `[p_total_norm, dP/dt]`
  - Binarisierung der Target-Geräte mittels Leistungsschwelle `on_w`
  - Fensterung als Seq2Seq-Datensatz (Fensterlänge `window`, Schrittweite `stride`)
- Modell
  - GRU über Zeitfenster
  - Output: Logits pro Zeitschritt und Gerät `(B, T, D)`
- Training/Evaluation
  - Zeitlicher Split: 70 % / 15 % / 15 % (Train/Val/Test)
  - Loss: `BCEWithLogitsLoss` mit `pos_weight` pro Gerät (Klassenimbalance)
  - "Fruehes" stoppen auf Macro-F1 (Validierung)
  - Schwellenoptimierung `τ*` pro Gerät: maximiert F1 auf dem Validierungssplit
  - Test-KPIs auf Basis von `τ*` (Konfusion, Precision/Recall/F1, Macro-F1)
- Reporting
  - JSON-KPIs + CSV Metriken pro Geraete
  - Plots: Konfusionsmatrizen, Balken (P/R/F1), F1(τ)-Kurven, Timelines

## Voraussetzungen

- Python `>= 3.10`
- Installierte Abhängigkeiten (siehe `pyproject.toml`)
- Optional für DEDDIAG:
  - PostgreSQL-Zugriff und vorhandene DB-Funktion `get_measurements(...)` im gewünschten Schema
  - Optionales Paket `deddiag` (enthält `psycopg[binary]`)
- Optional für GPU-Training:
  - CUDA-fähige PyTorch-Installation und passende Treiber

## Installation

Empfohlen ist eine venv-Installation im Editable-Modus:

python -m venv .venv
Windows: .venv\Scripts\activate
Linux/Mac: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .

## Training

python -m hems_nilm_gateway.training.train_mgru --config training/configs/config.yaml

## Artefakte und Outputs

- Pro Run wird ein Zeitstempel-Unterordner erzeugt: artifacts/out_dir/<artifact_prefix>_YYYY-MM-DD_HHMMSS/
  - Darin werden gespeichert:
    - model.pt: PyTorch State Dict des besten Modells
    - config.yaml: Kopie der verwendeten Konfiguration
    - normalizer.json: Z-Score-Parameter (mean/std) aus dem Trainingssplit
    - kpis.json: Test-KPIs inkl. τ* je Gerät und Konfusionszahlen
    - per_device_metrics.csv: Tabellarische Kennzahlen je Gerät (τ, P/R/F1, TP/FP/FN/TN)
    - summary.png: Zusammenfassungsplot (Konfusion, Balken, F1(τ)-Kurven)
    - timelines.png: Zeitabschnitte (Wahrheitszustaende vs. Score vs. Pred je Gerät) (Pred = Vorhersage Zustaende)

## Hinweise

- Dieses Repository ist Bestandteil einer Masterthesis. Eine Weiterverwendung außerhalb des Thesis-Kontexts ist nicht vorgesehen.
