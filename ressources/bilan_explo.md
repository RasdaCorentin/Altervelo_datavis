● Voici le bilan complet de l'analyse exploratoire.

  ---
  Bilan de l'analyse exploratoire — AlterVélo Réunion

  ---
  Hypothèses confirmées

  Vélos fantômes (§5 NB2) — confirmation nette.
  L'hypothèse était : certains vélos remontés par l'API ne sont plus en service réel. Résultat : 31
  vélos sur 128 (24.2% de la flotte) n'ont jamais été rechargés sur 28 jours → fantômes certifiés. Ils
  représentent 25.3% des lignes brutes. Plus important : 37.3% seulement d'entre eux sont marqués
  is_disabled — le filtre naïf laissait donc passer 63% des fantômes dans le dataset. C'est la
  justification technique principale de clean_vehicle_status.py et du critère temporel "jamais
  rechargé".

  Hétérogénéité du réseau (§7 NB1) — confirmée.
  Le ratio std des stations va de 2.76 (Port) à 0.00 (Mairie du Ouaki). Plusieurs stations ont une
  moyenne quasi-nulle (Kerveguen 0.06, Cité Palissade 0.04, Roches Maigres 0.00). Prédire ces stations
  est trivial — et inutile. La valeur ajoutée d'un modèle est concentrée sur les 8-10 stations à forte
  rotation.

  Sanity check jointure (§6 NB2) — validé.
  89.7% des paires (timestamp, station) ont une différence exacte de 0 entre le comptage de vélos actifs
   et num_vehicles_available. 98.6% ont un écart ≤ 1. La fusion vehicles → stations est techniquement
  propre.

  Transit corrélé négativement avec la disponibilité (§4 NB2) — confirmé.
  Corrélation -0.596 : quand les vélos en transit augmentent, la disponibilité en station baisse. Le
  signal est présent, même si la relation n'est pas parfaite.

  ---
  Fausses routes et surprises

  Anti-corrélation vélos ↔ docks (§4 NB1) — hypothesis erronée.
  L'attendu était une corrélation ≈ -1. La réalité : num_docks_available agrégé sur le réseau est
  constant à 229 (std = 0) sur tous les snapshots. Le résultat est NaN. Cela ne dit pas que les docks
  sont tous libres en permanence — cela dit que le total réseau est figé, probablement parce que les
  docks désactivés (num_docks_disabled = 0 partout, colonne constante) ne sont jamais renseignés. Ce
  champ est inutilisable au niveau réseau. Au niveau station individuelle la corrélation existe bien

  Vélos fantômes (§5 NB2) — confirmation nette.
  L'hypothèse était : certains vélos remontés par l'API ne sont plus en service réel. Résultat : 31 vélos sur 128 (24.2% de la flotte) n'ont jamais été rechargés sur 28 jours → fantômes certifiés. Ils représentent 25.3% des lignes
  brutes. Plus important : 37.3% seulement d'entre eux sont marqués is_disabled — le filtre naïf laissait donc passer 63% des fantômes dans le dataset. C'est la justification technique principale de clean_vehicle_status.py et du
  critère temporel "jamais rechargé".

  Hétérogénéité du réseau (§7 NB1) — confirmée.
  Le ratio std des stations va de 2.76 (Port) à 0.00 (Mairie du Ouaki). Plusieurs stations ont une moyenne quasi-nulle (Kerveguen 0.06, Cité Palissade 0.04, Roches Maigres 0.00). Prédire ces stations est trivial — et inutile. La
  valeur ajoutée d'un modèle est concentrée sur les 8-10 stations à forte rotation.

  Sanity check jointure (§6 NB2) — validé.
  89.7% des paires (timestamp, station) ont une différence exacte de 0 entre le comptage de vélos actifs et num_vehicles_available. 98.6% ont un écart ≤ 1. La fusion vehicles → stations est techniquement propre.

  Transit corrélé négativement avec la disponibilité (§4 NB2) — confirmé.
  Corrélation -0.596 : quand les vélos en transit augmentent, la disponibilité en station baisse. Le signal est présent, même si la relation n'est pas parfaite.

  ---
  Fausses routes et surprises

  Anti-corrélation vélos ↔ docks (§4 NB1) — hypothesis erronée.
  L'attendu était une corrélation ≈ -1. La réalité : num_docks_available agrégé sur le réseau est constant à 229 (std = 0) sur tous les snapshots. Le résultat est NaN. Cela ne dit pas que les docks sont tous libres en permanence —
  cela dit que le total réseau est figé, probablement parce que les docks désactivés (num_docks_disabled = 0 partout, colonne constante) ne sont jamais renseignés. Ce champ est inutilisable au niveau réseau. Au niveau station
  individuelle la corrélation existe bien (c'est une contrainte capacitaire : vélos + docks = capacité), mais elle est triviale et n'apporte rien.

  Pic heures de pointe en transit (§4 NB2) — absent.
  L'hypothèse était : pics à 7-9h et 17-19h. La réalité : la courbe de transit est quasi-plate à ≈ 30 vélos toute la journée, avec un léger creux en milieu de matinée (28.6 à 7h) et un très léger pic en soirée (30.8 à 19h). Le delta
  max sur 24h est de 2.2 vélos. Il n'y a pas de comportement domicile-travail lisible dans ces données — ce qui peut s'expliquer par la géographie de La Réunion (réseau diffus, usage loisir/tourisme fort) ou simplement par le trop
  faible volume de données pour faire apparaître le signal.

  Cycle hebdomadaire (§9 NB1) — quasi-absent.
  ACF à lag 336 (1 semaine) = 0.04. C'est statistiquement négligeable. La feature lag_336 qui semblait prometteuse ne porte presque aucun signal dans cette fenêtre de 28 jours. Le cycle journalier existe (ACF lag 48 = 0.13) mais il
  est lui-même faible.

  Profil journalier (§5 NB1) — signal microscopique.
  Le delta max weekday/weekend est de +0.10 vélos à 8h et -0.08 à 18h. Sur une capacité de 14 vélos en moyenne par station, c'est un effet de moins de 1%. La différence existe en direction mais elle est sans signification pratique à
  cette résolution temporelle.

  ---
  Bilan du nettoyage de données

  ┌───────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │             Problème              │                                               Découverte                                                │                                       Action                                       │
  ├───────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 30.9% d'imputation                │ Uniforme sur toutes les stations (pas variable) — indique un gap systématique de collecte, pas des      │ Imputation forward-fill acceptable mais le modèle "voit" beaucoup de données       │
  │                                   │ pannes isolées                                                                                          │ synthétiques                                                                       │
  ├───────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ num_docks_disabled = 0 toujours   │ Colonne constante, inutilisable                                                                         │ À exclure des features                                                             │
  ├───────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ num_vehicles_disabled = 0.014     │ Quasi-constante, très peu de signal                                                                     │ À exclure ou traiter avec précaution                                               │
  │ moyen                             │                                                                                                         │                                                                                    │
  ├───────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 31 vélos fantômes sur 128         │ Non détectables via is_disabled seul                                                                    │ Critère "jamais rechargé" → clean_vehicle_status.py                                │
  ├───────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Station 4 : écart jointure -0.805 │ La plus grande anomalie de jointure                                                                     │ À surveiller dans les features véhicules                                           │
  └───────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Pourquoi battre la MAE persistante sera difficile

  La MAE persistance est de 0.087 vélo. Le meilleur XGBoost stations-only était à 0.17-0.18, soit le double. Ce n'est pas un bug de modèle — c'est une conséquence directe de la structure des données.

  Le problème de fond : 28 jours de données à granularité 30 minutes.

  La heatmap (§6 NB1) l'illustre le mieux. Pour chaque cellule (jour de la semaine, heure), on dispose d'environ 3 à 4 mesures. Ce tableau est la "connaissance" que le modèle a du comportement d'un Lundi à 8h. Avec 3 observations, on
  ne peut pas distinguer un pattern réel d'une fluctuation aléatoire.

  La persistance (préditre t+1 = t) ne souffre pas de ce problème : elle utilise la valeur la plus récente disponible, qui est structurellement très proche de la valeur suivante puisque l'ACF à lag 1 (30 min) est de 0.96. Autrement
  dit, la série change très peu d'un pas de temps à l'autre. Un modèle doit apporter quelque chose au-delà de cette inertie naturelle — et pour cela il lui faut identifier des exceptions (les moments où la valeur va changer
  brusquement). Détecter ces exceptions fiablement requiert d'avoir vu des dizaines de mardi matin, pas trois.

  Pour donner un ordre de grandeur : un cycle annuel complet avec cette granularité représenterait ~17,500 timestamps par station. Avec 28 jours on en a 1,340. On travaille avec 7.6% d'une année. La "connaissance" du modèle est donc
  extrêmement fragmentaire, et tout pattern extrait est susceptible de sur-ajustement sur les quelques semaines observées (notamment la semaine de Pâques qui est peut-être présente et n'est pas représentative d'une semaine ordinaire).

  Le scénario réaliste avec les données actuelles : un modèle bien réglé pourrait approcher la MAE persistante, pas la battre de façon robuste. La valeur des features véhicules (transit, batterie) sera de réduire légèrement l'erreur
  sur les stations à forte rotation, là où la persistance échoue lors des changements brusques — mais les données restent trop courtes pour que l'amélioration soit statistiquement solide.
