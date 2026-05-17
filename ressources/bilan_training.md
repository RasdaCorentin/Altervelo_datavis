# Bilan — Entraînement des modèles AlterVélo Réunion

---

## 1. Pipeline de données

Le pipeline transforme les données brutes de l'API GBFS AlterVélo en entrées prêtes pour l'entraînement en cinq étapes séquentielles.

```
[API AlterVélo]
      │
      ├── station_status.csv       ──► cleaning_data.py      ──► stations_clean.csv
      │
      └── vehicle_status.csv       ──► clean_vehicle_status.py ──► vehicles_clean.csv
                                                                         │
                                   merge_csv.py ◄────────────────────────┤
                                        │                                │
                                        ▼                                ▼
                                stations_enriched.csv     build_vehicle_flow.py
                                                                   │
                                                                   ▼
                                                          vehicle_flow.csv
```

**`cleaning_data.py`** — Ingestion des statuts stations. Traduit les `station_id` opaques en index entiers, construit une grille temporelle à pas fixe de 30 minutes, applique une interpolation linéaire de jour et un forward-fill de nuit (20h–5h UTC+4), puis localise les timestamps en UTC+4 (Réunion).

**`clean_vehicle_status.py`** — Même quantification 30 min côté vélos. Filtre les vélos fantômes (batterie à 0 % sur toute la fenêtre d'observation) et recadre la fenêtre temporelle sur celle de `stations_clean.csv`.

**`merge_csv.py`** — Fusionne les deux sources : agrégats à la borne (état de la flotte présente) et histogramme spatial des vélos en transit par bandes concentriques de 150 m (0–900 m). Produit `stations_enriched.csv`.

**`build_vehicle_flow.py`** — Calcule, pour chaque couple (station, timestamp), les arrivées et départs de vélos dans les bandes 0–150 m et 150–300 m sur des fenêtres de 30, 60 et 120 min, plus les flux larges (300–600 m) à 60 min, soit **22 features de flux** + 2 ratios densité-pondérés. Produit `vehicle_flow.csv`.

### Qualité des données actuelles

La collecte porte sur environ **22 jours** d'historique, soit ~978 timestamps possibles par vélo. Les trous sont comblés par imputation (flag `is_imputed`). Le modèle voit donc un signal partiellement synthétique, ce qui plafonne mécaniquement ses performances.

---

## 2. Modèle Naïf (`train_naive.py`)

### Objectif

Établir une **baseline honnête** : un modèle XGBoost sans feature engineering, entraîné à prédire la valeur absolue `num_vehicles_available` à t+1h, sans fuite de données.

> La version précédente utilisait `y = valeur courante`, ce qui donnait une MAE ≈ 0 par fuite totale (le modèle lisait la réponse). La version corrigée prédit `y = valeur à t+horizon`.

### Résultats (horizon 60 min, 41 478 lignes)

| Métrique | Modèle naïf | Persistance |
|---|---|---|
| MAE | 0.165 | **0.086** |
| RMSE | 0.413 | **0.354** |

**Le modèle naïf ne bat pas la persistance.** Sans feature engineering, XGBoost concentre 77 % de son importance sur `num_vehicles_available` (la valeur courante) mais la prédit moins bien que de la recopier telle quelle. Cela confirme que les features brutes du CSV ne portent pas suffisamment de signal prédictif sans transformation.

### Diagnostic

L'importance des features révèle que le modèle ne dispose d'aucune information sur la dynamique temporelle ni sur le voisinage spatial. Il ne peut qu'approximer la persistance de façon dégradée.

---

## 3. Modèle Delta v4 (`train_xgboostplus_delta_v3.py`)

### Motivation

Sur la cible brute `y`, XGBoost dépense l'essentiel de son importance sur `current_value` + `lag_1` : le stock change peu en 1 h, donc recopier le présent est quasi-optimal. Les features de flux sont alors ignorées.

**Solution : prédire la variation** `Δy = y(t+h) − y(t)` au lieu de `y(t+h)`. La persistance devient l'hypothèse `Δy = 0`, et les features de flux deviennent les seules à porter le signal de changement.

### Architecture des features (68 au total)

| Groupe | Features | Rôle |
|---|---|---|
| Temporel cyclique | `hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `time_regime` | Rythmes quotidiens et hebdomadaires |
| Lags | `lag_1/2/4/6/12/48` | Historique court à long terme (30 min → 24 h) |
| Rolling | `roll_mean/std_4/12/48` | Tendance et volatilité locale |
| Diffs | `diff_1/4/48` | Dynamique en cours |
| Station | `station_index` (catégoriel) | Comportement propre à chaque station |
| Flux | `n_arrivees/departs_*`, `net_flow_*`, ratios densité | Mouvements de vélos dans le voisinage spatial |
| Batterie | `pct_low_battery`, `mean/min/max_battery` | Disponibilité réelle de la flotte |

### Résultats — avec `vehicle_flow.csv`

| Horizon | MAE modèle | MAE persistance | Lift | Anticip `\|Δ\|≥1` | Anticip `\|Δ\|≥2` |
|---|---|---|---|---|---|
| 60 min | 0.069 | 0.069 | −0.1 % | +0.278 | +0.415 |
| 90 min | 0.098 | 0.098 | −0.0 % | +0.152 | +0.258 |
| **120 min** | **0.127** | **0.127** | **+0.0 %** | **+0.183** | **+0.247** |
| 150 min | 0.154 | 0.154 | −0.0 % | +0.141 | +0.202 |

### Apport des features de flux

| Métrique | Sans `vehicle_flow.csv` | Avec `vehicle_flow.csv` | Gain |
|---|---|---|---|
| MAE 60 min | 0.089 | **0.069** | −22 % |
| MAE 120 min | 0.147 | **0.127** | −14 % |
| Anticip `\|Δ\|≥2` à 60 min | +0.109 | **+0.415** | ×3.8 |

L'ajout des features de flux réduit la MAE de 14 à 22 % selon l'horizon et multiplie par ~4 la corrélation entre les variations prédites et les variations réelles sur les mouvements significatifs (`|Δ| ≥ 2 vélos`).

---

## Conclusion — L'effet du temps sur les performances

Le résultat le plus important n'est pas le lift actuel (marginal) mais sa **direction** : avec seulement ~22 jours de données collectées, une partie significative du signal est synthétique (imputation) et le modèle ne dispose pas encore d'une base suffisante pour modéliser les cycles hebdomadaires et les événements atypiques.

Deux phénomènes s'améliorent mécaniquement avec le temps :

**1. Moins de données imputées.** Chaque jour supplémentaire de collecte réduit la proportion de lignes synthétiques. Les features de flux (`n_arrivees_*`, `net_flow_*`) sont particulièrement sensibles à la qualité des observations véhicule : un trou dans `vehicles_clean.csv` se propage directement en un flux nul fictif.

**2. Meilleure couverture des cycles longs.** Les features `lag_48` (24 h), `diff_48` et `roll_*_48` qui encodent le cycle journalier ne sont statistiquement stables qu'après plusieurs semaines d'historique. Les cycles hebdomadaires (`dow_sin/cos`) nécessitent au moins 3 à 4 semaines pour être appris correctement.

**Entre 22 jours et 1 mois de collecte, le modèle commence à battre la baseline à l'horizon 120 min** (lift +0.011 % déjà visible), même sur des données partiellement imputées. Ce seuil correspond au moment où les cycles longs deviennent exploitables et où les features de flux cessent d'être dominées par le bruit des trous. Au-delà d'un mois, on peut s'attendre à un lift positif et stable sur les horizons 60–120 min, en particulier sur les stations à forte variabilité (`Port`, `Front De Mer`, `Stade de Terre Sainte`) qui sont précisément celles où la persistance échoue le plus.
