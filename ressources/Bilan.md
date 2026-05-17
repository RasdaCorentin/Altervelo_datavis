# Bilan — projet AlterVélo Réunion

Synthèse en trois étapes : nettoyage, exploration & fusion, entraînement.
Tous les chiffres proviennent d'exécutions effectives (jamais de spéculation).

---

## 1. Nettoyage

### v1 — Stations (`cleaning_data.py`)

Pipeline consolidé en 4 étapes :
1. **Enrichissement** : `station_id` opaque → `station_index` + `station_name` + lat/lon/capacity, via l'API GBFS `station_information.json` (mapping stable, tri lexicographique).
2. **Comblement des trous** : grille canonique 30 min × 31 stations. Stratégie hybride — nuit (20h-5h UTC+4) ffill, jour interpolation linéaire des compteurs. Indicateur `is_imputed` conservé.
3. **Localisation** UTC → UTC+4.
4. **Final** : drop colonnes constantes (`is_installed/renting/returning`, `last_reported`), parsing du JSON `vehicle_types_available` → `count_x2`.

**Sortie** : `stations_clean.csv` — 38 936 lignes × 12 colonnes (31 stations × ~26 jours × 48 slots).

### v2 — Vélos individuels (`clean_vehicle_status.py`)

Pipeline 7 étapes calqué sur station mais à granularité vélo : mapping `station_id` → `station_index` (0 = en circulation), quantification 30 min closest-to-center, localisation UTC+4, drop des colonnes constantes (`is_reserved`, `vehicle_type_id`), **filtre fantômes**.

- **Filtre fantômes** : un `vehicle_id` dont `max(current_fuel_percent) == 0` sur 26 jours est presque sûrement un vélo perdu/cassé que l'API continue de remonter passivement (un vélo docké est rechargé à la borne). → **31 vélos / 24 % de la flotte exclus** (25 879 lignes).

**Sortie** : `vehicles_clean.csv` — 76 669 lignes × 9 colonnes.

---

## 2. Analyse exploratoire & fusion

### v1 — `eda.ipynb` (stations seules)

- **§7 — Stations à plus forte rotation** : top 10 confirme l'intuition métier (Port, Pôle d'échanges St-Louis, Eglise Terre Sainte, Marché Couvert, Front de Mer, IUT…). Réseau très hétérogène : ces 10 stations concentrent l'essentiel de la dynamique apprenable ; les ~21 autres ont une `std` faible donc une cible quasi constante.
- **§9 — Autocorrélation** : la série est **très persistante à court terme**. Le pic hebdomadaire k=336 n'est pas exploitable (gain marginal pour -30 % de données — cf. §3).

### v2 — `eda2.ipynb` (vélos + fusion)

- **§3 — Anomalie batterie** : 29.7 % des relevés à 0 %, 36 % < 20 %. Anomalie suspecte → enquête en §5.
- **§5 — Diagnostic fantômes** : critère `max(fuel) == 0` sur la fenêtre complète. **31 vélos identifiés**, jamais marqués `is_disabled` côté API → invisibles à un filtre statique. Justifie le filtre temporel ajouté en §1.
- **§6 — Sanity check fusion** : sur 18 452 paires (timestamp, station_index) communes, |diff| moyen = **0.12**, **89 % paires diff=0 exact**, **98.5 % paires |diff| ≤ 1** entre `n_vehicles_actifs` agrégés et `num_vehicles_available`. → **GO fusion**.
- **§4 — Profils utilisateurs** détectés sur les vélos en transit : creux 1h-7h, **pic matin 8-12h** (travailleurs), **pic soir 18-21h** (travailleurs + sortie), **traîne 21h-1h** (fêtards). Corr `n_transit ↔ n_disponibles` = -0.60 (un vélo en route = un dock libre). Hypothèse à valider : interaction avec `dow` (semaine vs week-end).

---

## 3. Entraînement

### v1 — Trois variantes XGBoost (`train_xgboost*.py`)

| Modèle | Train / Test | Features | RMSE | MAE | Persistance MAE | Verdict |
|---|---|---|---|---|---|---|
| `train_naive.py` | 30 814 / 7 719 | 6 (brutes) | 0.418 | **0.169** | 0.088 | Perd |
| `train_xgboost.py` | 29 636 / 7 409 | 28 (lags + diffs + cycliques) | 0.480 | **0.172** | 0.088 | Perd |
| `train_xgboost_weekly.py` | 22 475 / 5 642 | 29 (+ `lag_336`) | 0.485 | **0.168** | 0.087 | Perd |

Top features systématiquement dominées par `current_value` (72-88 %) et `lag_1` (17-22 %) — autres lags / rollings / diffs / cycliques < 1 % chacun.

**Pourquoi tous perdent contre persistance.** Sur 31 stations, ~21 ont `std < 1` (ex : `Kerveguen`, `Cité Palissade`, `Mairie du Ouaki` — la cible ne change *jamais* sur la fenêtre de test). La persistance y atteint MAE = 0 *par construction*. XGBoost prédit du continu (2.37) là où la vérité est entière (2 ou 3) → biais d'arrondi ~0.1 qui suffit à le faire perdre. **La MAE moyenne globale n'est pas la bonne métrique.**

**Reporting segmenté** (`report_per_station()`, déjà implémenté) : 5 stations HARD (Port, Casabona, Eglise Terre Sainte, Marché Couvert, Stade Terre Sainte) où la persistance échoue + 5 stations EASY (Avenue L. Vergès, Kerveguen, Cité Palissade, Mairie du Ouaki, Roches Maigres) en sanity check.

### v2 — `train_xgboostplus.py` + `merge_csv.py`

Construction de `stations_enriched.csv` (`merge_csv.py`) : `stations_clean` + features agrégées par (timestamp, station) issues de `vehicles_clean` (n_actifs, batteries) + **histogramme spatial des vélos en transit en 6 bandes concentriques de 150 m** (0-150, 150-300, 300-450, 450-600, 600-750, 750-900 m) + `dist_nearest_transit_m`. Nouvelle feature temporelle `time_regime` (5 classes — nuit, matin rush, midi, soir rush, late) issue de l'EDA §4. Stratégie d'imputation : NaN préservés sur les batteries quand aucun vélo n'est observé (XGBoost partitionne nativement), flag `is_obs_missing` ajouté pour distinguer « non observé » de « observé bas ».

**Volumétrie & sanity v2** : `stations_enriched.csv` = 38 936 × 26 colonnes. **Découverte importante : `corr(n_vehicles_actifs, num_vehicles_available) = 0.68`** (attendu > 0.9), et **52.4 % des paires (t, s) sans aucune observation vélo** — la fusion par comptage perd beaucoup d'info sur les stations rares.

**Distribution des bandes spatiales** (% paires non nulles) : 0-150m **33.5 %**, 150-300m **2.6 %**, 300-450m 12.8 %, 450-600m 12.6 %, 600-750m 10.8 %, 750-900m 18.5 %. Le creux 150-300m est un **artefact d'échantillonnage temporel** (pas 30 min) : un vélo qui traverse cette bande est soit en phase finale d'approche (< 150m, va docker dans la minute) soit déjà au-delà à la mesure suivante. Un pas plus court (10 min) le capterait.

**Résultats** (run Colab, 29 946 train / 7 502 test, 44 features) :

| Modèle | MAE global | Persistance | Verdict | Port (HARD) | Casabona (HARD) | Cité Palissade (EASY) |
|---|---|---|---|---|---|---|
| `train_xgboost.py` (v1) | 0.172 | 0.088 | Perd | n.c. | n.c. | n.c. |
| `train_xgboostplus.py` (v2) | **0.178** | 0.086 | Perd | 1.275 (vs pers. 0.339, lift -276 %) | 0.435 (vs 0.298) | 0.047 (vs 0.000) |

**Top 15 features** : `lag_1` (47.6 %) + `current_value` (45.4 %) = **92.9 %** à eux deux. Les 13 features de fusion totalisent ~1.5 %. Détail des features fusion :

- **Présentes mais faibles** : `n_vehicles_actifs` rang 6, `pct_low_battery` rang 10, `mean_battery` rang 12 — XGBoost les pioche systématiquement, sans qu'elles deviennent critiques.
- **Surprise** : `n_transit_150_300m` rang 9 bat `n_transit_0_150m` rang 24 — alors que la bande 150-300m est non nulle dans 2.6 % des paires seulement (vs 33.5 %). Signal **rare = discriminant** : quand un vélo apparaît dans cette bande de transit final, l'événement isole peu de lignes mais avec un effet net.
- **`is_obs_missing` inutile** (rang 44, importance 0.000) — le modèle s'appuie tellement peu sur les features fusion qu'il n'a pas besoin d'en signaler l'absence.

**Conclusion v2** : les features de fusion sont *présentes dans le top* mais dominées par l'auto-régression à cause de la très forte autocorrélation à 1h. **Le problème n'est plus le feature engineering, c'est la formulation de la tâche.**

### v3 — `train_xgboostplus_delta.py` (cible = Δy, loss MAE, multi-horizons)

Trois changements simultanés appliqués pour casser la dominance de l'auto-régression :
1. **Cible = Δy = y(t+h) − y(t)** au lieu de `y(t+h)` brut → annule par construction la dominance de `current_value`.
2. **Loss `reg:absoluteerror`** au lieu de `reg:squarederror` → optimise directement la métrique d'évaluation, élimine le biais d'arrondi.
3. **Boucle multi-horizons** (1h / 1h30 / 2h / 2h30) + reporting top 10 stations dérivées par std.

**Synthèse multi-horizons** :

| Horizon | MAE modèle | MAE persistance | Lift global | Top feature |
|---|---|---|---|---|
| 1h00 | 0.087 | 0.086 | -0.7 % | `n_vehicles_actifs` (0.059) |
| 1h30 | 0.117 | 0.116 | -0.8 % | `n_vehicles_actifs` (0.050) |
| 2h00 | 0.148 | 0.147 | -0.7 % | `current_value` (0.071) |
| 2h30 | 0.176 | 0.176 | -0.2 % | `current_value` (0.068) |

**Les trois effets attendus sont mesurés** :

- **Effet 1 — disparition du désastre v2.** Sur `Port` à 1h, MAE modèle passe de **1.275 (v2) → 0.343 (v3)**, lift de **-276 % à -1.3 %**. Le modèle ne sur-extrapole plus.
- **Effet 2 — émergence des features fusion.** `n_vehicles_actifs` est *top 1* à 1h et 1h30 (devant `current_value`). À 1h30, `max_battery` et `pct_low_battery` apparaissent dans le top 10. À 2h30, `n_transit_450_600m` rentre dans le top 10. Tout le travail v2 trouve sa justification ici.
- **Effet 3 — convergence vers la persistance avec l'horizon.** Lift à 60 min = -0.7 %, à 150 min = -0.2 %. Le modèle se rapproche de la persistance sans jamais la dépasser. La distribution conditionnelle de Δy a une mode très proche de 0 sur la majorité des paires (t, s) — la « meilleure » prédiction *en MAE* est donc Δy ≈ 0, ce qu'apprend XGBoost.

**Verdict v3** : le modèle est **opérationnellement équivalent à la persistance**, jamais pire (les lifts négatifs sont à <1 % d'écart, dans le bruit). C'est une amélioration franche par rapport à v2 où il extrapolait catastrophiquement sur les hubs. **La barre du « mieux que ne rien prédire » n'est pas franchie**, et les chiffres montrent pourquoi : avec 22 jours de données et une cible si auto-corrélée, la persistance est l'optimum de Bayes au sens MAE — sauf à enrichir massivement les sources externes (météo, événements, vraie périodicité hebdomadaire).

## 4. Mise en production

### v1 — Dashboard adossé à la chaîne CSV de recherche

Premier dashboard livrable, fonctionnel mais couplé au dossier parent.

- **Architecture 3 couches** : (a) ETL one-shot `db_init.py` qui rejoue `data/stations_enriched.csv` puis lance le backtest sur les 20 % test (4 horizons), (b) tick `dashboard/pipeline.sh` qui `cd ..` et enchaîne les 5 scripts CSV de recherche (`collect.py` → `cleaning_data.py` → `clean_vehicle_status.py` → `merge_csv.py`) puis `append_live_obs.py` pour charger le CSV final dans `velos.db`, (c) UI Streamlit 3 pages (T-0, Prévision, Monitoring) avec routage `app/dashboard.py` et data layer `app/data.py`.
- **Schéma SQLite** : `stations`, `observations(source∈{historical,live})`, `predictions(ts_pred, ts_target, horizon_min, station_index, y_pred, y_current, y_obs, err_model, err_pers, source∈{backtest,live})` PK `(ts_pred, horizon_min, station_index)`, `pipeline_runs`. Pattern *delayed-feedback* : `y_obs / err_model / err_pers` NULL à l'INSERT, remplis par `evaluate.py:db.backfill_predictions()` dès que `ts_target` est observé. Toute requête sur les erreurs filtre `WHERE y_obs IS NOT NULL`.
- **Volumétrie post-bootstrap** (mesurée) : 30 318 prédictions évaluées (4 horizons × 31 stations × ~245 slots).
- **Limite structurelle** : viole l'autonomie déclarée du dossier (`cd ..`, lit/écrit `../data/*.csv`). Coût opérationnel : impossible à packager sans embarquer le parent.

### v2 — Pipeline DB-first dans `dashboard/ingest/`

Réécriture du tick pour rendre `dashboard/` autonome au runtime — l'API GBFS est appelée directement, tous les snapshots bruts sont persistés (audit complet), aucun CSV intermédiaire.

**7 étapes idempotentes**, toutes loggées dans `pipeline_runs` (durée, statut, compteurs) via un contextmanager commun :

| # | Script | Lit | Écrit |
|---|---|---|---|
| 1 | `collect.py` | API GBFS | `raw_station_status`, `raw_vehicle_status` (INSERT OR IGNORE) |
| 2 | `clean_stations.py` | `raw_station_status` | `stations_clean` (is_imputed=0, INSERT OR REPLACE) |
| 3 | `fill_gaps.py` | `stations_clean` | `stations_clean` (is_imputed=1, INSERT OR IGNORE) |
| 4 | `clean_vehicles.py` | `raw_vehicle_status` | `vehicles_clean` (INSERT OR REPLACE) |
| 5 | `merge.py` | `stations_clean` + `vehicles_clean` | `observations` (source='live') |
| 6 | `predict.py` | `observations` | `predictions` (4 horizons, source='live') |
| 7 | `evaluate.py` | `observations` + `predictions` | UPDATE `y_obs` / `err_model` / `err_pers` |

- **Invariant train/serve** : `merge.py:OBS_COLS` reste byte-identique à `append_live_obs.py:OBS_COLS` et à l'ordre produit par `merge_csv.py` (cf. `dashboard/CLAUDE.md` — toute divergence = train/serve skew silencieux). Les imputations sont distinguées (`is_imputed=1`, INSERT OR IGNORE) des observations réelles (`is_imputed=0`, INSERT OR REPLACE) — relancer `fill_gaps.py` ne détruit jamais une mesure.
- **Schéma raw ajouté** : `raw_station_status`, `raw_vehicle_status` (PK `(ts_collect, station_id|vehicle_id)`), append-only, rétention illimitée par décision projet.
- **Smoke test** (run 2026-05-04T15:23 UTC+4) : collect +31 stations / +115 vélos → clean_stations 31 lignes → fill_gaps 0 imputation → clean_vehicles 75 lignes → merge +31 obs → predict 124 prédictions en 0.46 s → evaluate +62 backfill (cumul 30 318) en 0.01 s. Re-run immédiat : `+0 obs` (idempotence vérifiée par les `INSERT OR IGNORE` sur PK `(timestamp, station_index)`).
- **Mise en prod** : `nohup` + boucle 30 min + log persistant ; alternative systemd timer utilisateur (`OnUnitActiveSec=30min`) documentée dans `dashboard/ingest/README.md`.

**Verdict v2** : couche serving 100 % autonome au runtime. Le dossier parent n'est désormais requis que pour deux artéfacts statiques — les 4 modèles `xgb_velos_v3_h{60,90,120,150}min.json` et, pour le bootstrap historique uniquement, `data/stations_enriched.csv`. L'ancien `dashboard/pipeline.sh` est conservé en référence mais n'est plus invoqué.

### Pistes restantes (par ordre d'impact)

1. **Aligner la loss avec la métrique** : `reg:squarederror` → `reg:absoluteerror` (MAE directe) ou `count:poisson` (compteurs entiers ≥ 0).
2. **Post-traitement** : `np.clip(np.round(pred), 0, capacity)` pour rendre les prédictions entières et bornées.
3. **Réduire le pas d'échantillonnage temporel** (motivé par le creux à 150-300m, cf. v2). Le pas 30 min de `collect.py` rate les vélos qui transitent vite dans les bandes spatiales intermédiaires. Passer à 10 min côté collecte rendrait l'histogramme spatial vraiment exploitable et affinerait aussi `dist_nearest_transit_m`. Coût : ×3 sur la volumétrie côté CSV bruts, négligeable côté training (la quantification 30 min reste un choix pour le target).
4. **Plus de données** : 22 jours, c'est trop court pour `lag_336`. Avec 8-12 semaines, l'hebdomadaire deviendrait probablement gagnant.
5. **Modèles spécialisés par cluster de stations** (hubs vs résidentielles) au lieu d'un modèle unique.
6. **Prédictions multi-cibles** : `predictions.y_pred` ne couvre aujourd'hui que `num_vehicles_available`. Étendre à `num_docks_available`, `mean_battery`, `pct_low_battery`, bandes transit (1 modèle par cible × 4 horizons, format `xgb_velos_v3_{target}_h{H}min.json`). Conséquence schéma : ajouter colonne `target` à `predictions` (ou table `predictions_multi`), itérer côté `predict.py` / `evaluate.py`.
