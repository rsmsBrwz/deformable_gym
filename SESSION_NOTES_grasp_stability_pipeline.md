# Session Notes: Grasp-Stability-Measures & Trainings-Pipeline für `MjFloatingMiaGraspBoxes-v0`

**Datum:** 2026-07-03 bis 2026-07-05
**Scope:** `deformable_gym`, MuJoCo-Backend, Boxes-Objekt (`assets/objects/mjcf/boxes.xml`), `mia_hand`-Roboter, Stable-Baselines3 (PPO/SAC/TD3/A2C/DDPG)

## TL;DR

Ziel war ein Trainings-/Eval-Setup für RL-Grasping auf einem neuen, ungetesteten Environment (deformierbare Boxen), inkl. Grasp-Stability-Measures und Reporting. Das Environment war beim ersten Testlauf für 4 von 5 SB3-Algorithmen praktisch unbrauchbar (Simulation destabilisierte fast sofort, ein Lauf hing sogar 11,5h ohne Fortschritt). Die vermutete Ursache (Elastizitätsmodell der Boxen) war **falsch** – nach systematischem Ausschlussverfahren stellte sich die tatsächliche Ursache als eine **kinematische Singularität (Gimbal-Lock)** in der Orientierungssteuerung der Hand heraus, kombiniert mit einem echten Ctrl-Akkumulations-Bug. Nach den Fixes liefen 3 von 5 Algorithmen über 1 Mio. Schritte mit 0 % Instabilität; die verbleibenden zwei zeigen jeweils einen anderen, genauer eingegrenzten Restfall.

---

## 1. Ausgangslage

- Environment `MjFloatingMiaGraspBoxes-v0` (Registrierung in `deformable_gym/__init__.py:register_mj_grasp_envs`) war zwar lauffähig, aber es existierten keinerlei Grasp-Stability-Measures, und die vorhandenen Trainingsskripte (`pipeline.py`, `train_test.py`) waren funktional defekt (`env.get_grasp_stability()`/`env.get_object_position()` existierten nicht; SB3-API-Fehlgebrauch in `train_test.py`).
- Anforderung: (a) Grasp-Stability-Measures pro Test-Episode speichern, (b) Trainingskennzahlen für Analyse sichern, (c) Parameter so wählen, dass ein Lauffähigkeitstest lokal in vertretbarer Zeit möglich ist.

## 2. Architektur (Kurzreferenz)

- `BaseMJEnv` → `GraspEnv` (`deformable_gym/envs/mujoco/`): Szene = Robot-MJCF + Objekt-MJCF + `mj_scene_base.xml`, gemergt via `asset_manager.create_scene`.
- Reward: sparse, nur am Episodenende (`_pause_simulation` löst Objekt-Fixierung, friert Robotergelenke ein, prüft nach 1 s Objekthöhe: `+1`/`-1`).
- Objekt `boxes`: 3 unabhängige `flexcomp`-Soft-Bodies (`box_left/center/right`), je 8 Vertex-Bodies mit 3 Slide-Joints (kein Rotations-DOF pro Vertex), gehalten durch `<edge equality="true">` während der Episode (Soft-Body verhält sich dadurch näherungsweise starr, bis am Episodenende gelöst).
- Roboter `mia_hand`: bei `control_type="joint"` steuert die Policy **sowohl** Finger-Aktuatoren **als auch** die 6 Basis-Pose-Aktuatoren (`ee_A_X/Y/Z/OX/OY/OZ`) – das war zunächst nicht offensichtlich und Ursache eines Analysefehlers (siehe 5.5).

## 3. Implementierte Grasp-Stability-Measures

Referenz: extern bereitgestellte Literaturübersicht (`grasp_stability_context_mujoco.md`), umgesetzt in `deformable_gym/helpers/grasp_metrics.py` (neues Modul) + Integration in `grasp_env.py`.

| Measure | Typ | Implementierung |
|---|---|---|
| Binary Grasp State | per-step | `binary_grasp_state()`, Schwelle auf Kontaktanzahl + Normalkraft |
| Contact Score | per-step | `contact_score()`: Kontaktanzahl + Summe/Mittel der Normalkraft via `mujoco.mj_contactForce` |
| Object Retained Ratio | episodisch | Anteil der Objekt-Teilkörper (`mju.get_direct_child_body_names`) mit aktivem Handkontakt |
| Energy-Based Quality | episodisch | `data.energy` (potential/kinetic), aktiviert via `model.opt.enableflags |= mjENBL_ENERGY`; grobe Näherung (mischt Gravitations- und Elastizitätsenergie) |
| Dynamic Stability Probe | episodisch | Kurzer externer Störimpuls (`data.xfrc_applied`) auf die DOF-tragenden Objekt-Vertex-Bodies während der Pause-Phase; misst max. COM-Verschiebung, Restgeschwindigkeit, `settle_step` (Schritte bis Geschwindigkeit unter Schwelle bleibt) |

**Technisches Detail Kontakterkennung:** MuJoCo-Kontakte zwischen starren Geoms und Flex-Elementen haben `geom2 == -1` und `contact.flex == [-1, flex_id]`. Da `flex_id` über `mju.id2name(model, flex_id, "flex")` auf denselben Namen wie der zugehörige Body auflöst, konnte eine generische, namenskonventions-unabhängige Zuordnung implementiert werden (`hand_object_contact_ids` in `grasp_metrics.py`, Body-Subtree-Traversierung via neuer Helper `mj_utils.get_body_subtree_ids`). Das war nötig, weil `shadow_hand`-Geoms in der Szene **unbenannt** sind – eine Filterung über Geom-Namens-Prefixes (ursprünglicher Plan) hätte nur für `mia_hand` funktioniert.

Alle episodischen Keys sind ab `reset()` mit `NaN` vorbelegt (`_EPISODE_METRIC_KEYS` in `grasp_env.py`), damit sie auch bei vorzeitigem Truncate (Instabilität) im Info-Dict vorhanden sind – Voraussetzung dafür, dass SB3s `Monitor(info_keywords=...)` nicht mit einem `AssertionError` abbricht.

Reward-Funktion wurde bewusst **nicht** verändert – die Measures sind rein zusätzliche Beobachtungsgrößen im Info-Dict.

## 4. Trainings-Pipeline (`pipeline.py`, Neufassung)

- CLI-konfigurierbar (`--envs --algorithms --seeds --total-timesteps --eval-freq --n-eval-episodes --max-episode-steps --idle-timeout-minutes --run-timeout-minutes`).
- Pro Run: SB3 `Monitor` (mit `info_keywords`) fürs Trainings-Log, eigener `GraspMetricsEvalCallback` für periodische deterministische Evaluation → `grasp_stability.csv`, SB3-Logger (`stdout/csv/tensorboard`) → `progress.csv`.
- **`NanSafeWrapper`** (`gym.Wrapper`): erkennt nicht-finite Observationen und erzwingt zusätzlich ein hartes Step-Limit pro Episode (Default 800, ≈ 2× Normalepisode) – Absicherung gegen den Fall, dass `data.time` nach einer Instabilität nicht mehr monoton läuft und die reguläre Terminierung (`sim_time >= max_sim_time`) nie greift.
- **Watchdog-Architektur** (`run_with_watchdog`, `run_single`, `HeartbeatCallback`): Jede (env, algo, seed)-Kombination läuft in einem eigenen `multiprocessing.Process` (Kontext `"spawn"`). Ein `HeartbeatCallback` meldet alle 200 Steps den Fortschritt über eine `Queue` an den Elternprozess. Zwei Limits:
  - `--idle-timeout-minutes` (Default 15): maximale Lücke zwischen zwei Heartbeats.
  - `--run-timeout-minutes` (Default 240): absolute Wall-Clock-Obergrenze pro Run.
  Bei Überschreitung wird der Subprozess per `terminate()`→`kill()` beendet; die Pipeline schreibt eine `timed_out=True`-Platzhalterzeile und macht mit der nächsten Kombination weiter. `summary.csv` wird nach **jedem** Run neu geschrieben (nicht erst am Ende), damit ein späterer Abbruch keine bereits fertigen Ergebnisse kostet.
  - **Warum ein Subprozess statt Python-Timeout/Signal:** Der ursprüngliche Incident (siehe 5.6) war ein einzelner `mj_step`-Aufruf, der nie zurückkehrte – ein reiner Python-seitiger Post-Step-Check (wie `NanSafeWrapper`) kann das nicht abfangen, da der Prozess innerhalb des nativen Aufrufs hängt. Nur ein externes `SIGKILL` auf den Prozess wirkt zuverlässig unabhängig vom internen Zustand.
- Zusätzliche Diagnose-Metriken im Info-/Eval-Pfad: `sim_unstable` (Episode via Truncation statt regulärer Terminierung beendet) und `action_saturation` (Anteil der Aktionsdimensionen nahe an ihren Grenzen, Schwelle 5 % der Range) – direkt motiviert durch die Instabilitäts-Untersuchung (Abschnitt 5).

## 5. Instabilitäts-Untersuchung – chronologisch

### Symptom
Bei ~1000 Timesteps Smoke-Test liefen alle 5 Algorithmen. Bei realistischerer Skala (200k–1M Timesteps) destabilisierte sich die Simulation reproduzierbar (`WARNING: Nan, Inf or huge value in QACC`), und in einem Fall hing der gesamte Prozess **11,5 Stunden** ohne jeden Fortschritt (kein Logging, 98 % CPU) – ein einzelner `mj_step`-Aufruf kehrte nie zurück, vermutlich weil eine nicht-finite Position die Broad-Phase-Kollisionserkennung kombinatorisch explodieren ließ.

### Reproduktion (Stresstest-Methodik)
Da echte RL-Trainingsläufe für Iteration zu langsam sind, wurde ein Stresstest-Script gebaut, das feste Aktionsmuster gegen `MjFloatingMiaGraspBoxes-v0` fährt und Instabilität über zwei Signale erkennt: (a) `np.isfinite(obs)` und (b) `data.time` monoton steigend (ein Wiederauftreten des 11,5h-Hangs zeigte sich als **nicht-monotones** `data.time` nach der Instabilitätswarnung – ein zweites, eigenständig entdecktes Symptom). Getestete Muster: `constant_strong` (0.6×max, konstant), `constant_moderate` (0.2×max), `biased_noise` (zufälliger Bias + Rauschen pro Episode, zur Nachbildung einer untrainierten Policy), `random_uniform`.

**Baseline:** 15/32 Fehler (47 %), praktisch 100 % bei `constant_strong`.

### Ausgeschlossene Hypothesen (mit jeweiligem Test)

| # | Hypothese | Test | Ergebnis |
|---|---|---|---|
| 1 | Elastizitätsmodell der Boxen zu weich/steif | `damping` 1e-4→1e-2 (100×), `young` unverändert getestet | **Kein Effekt** (bit-identische Fehlerzeitpunkte) |
| 2 | Integrator (explizit vs. implizit) | `model.opt.integrator` → `mjINT_IMPLICITFAST` zur Laufzeit gepatcht | **Kein Effekt** |
| 3 | `<edge equality>`-Constraint der Flexcomp | Constraint komplett entfernt | **Kein Effekt** |
| 4 | Kontakt-Solver zu steif | `mjENBL_OVERRIDE` + verschiedene `o_solref`/`o_solimp`/`o_margin`-Kombinationen (weicher Timeconst, weicheres Impedanzprofil, Margin) | **Kein/kaum Effekt** (13–14/16 statt 15/16) |
| 5 | Aktuatorkraft zu hoch (Finger pressen zu stark) | `model.actuator_forcerange` auf 50 %/25 %/10 % skaliert | **Kein Effekt** (8/8 Fehler bei allen Skalierungen) |

Wichtige methodische Randbemerkung: Ein Zwischentest, bei dem die 6 Basis-Pose-Aktuatoren "auf 0 gesetzt" wurden, um Finger-Selbstkollision isoliert zu testen, war **fehlerhaft konzipiert** – die Annahme "Indizes 0–5 = ee-Aktuatoren" stimmte nicht mit der tatsächlichen `robot.actuators`-Reihenfolge überein (`[thumb, index, ee_X, ee_Y, ee_Z, ee_OX, ee_OY, ee_OZ, ...]`); tatsächlich blieben `ee_OY`/`ee_OZ` aktiv. Der scheinbare Beleg für eine "separate Fingerkollisions-Ursache" löste sich nach Korrektur auf und war konsistent mit Hypothese 6.

### Root Cause: Kinematische Singularität (Gimbal-Lock)

Die Basis-Orientierungsaktuatoren `ee_A_OX/OY/OZ` (3 sequentielle Hinge-Joints, Euler-Winkel-artige Darstellung der Handorientierung) hatten `ctrlrange="-1.570796 1.570796"` (±90°). Empirischer Parametersweep (`model.actuator_ctrlrange` zur Laufzeit variiert):

| Limit (rad) | `constant_strong` | `biased_noise` |
|---|---|---|
| 1.5708 (Original) | 8/8 fail | 5/8 fail |
| 1.4 | 8/8 fail | 5/8 fail |
| 1.3 | 0/8 | 2/8 fail |
| 1.2 | 0/8 | 1/8 fail |
| 1.1 | 0/8 | 0/8 |

Scharfer Übergang zwischen 1.4 und 1.3 – klassisches Symptom einer Gimbal-Lock-Singularität nahe ±π/2 in einer 3-Hinge-Euler-Kette. **Interessanter Nebenbefund:** `shadow_hand` begrenzt dieselben Aktuatoren bereits auf `ctrlrange="-1 1"` (in `shadow_hand.xml`) – vermutlich deshalb war dieses Problem dort nie aufgefallen.

**Fix (initial):** `ctrlrange` in `deformable_gym/assets/robots/mjcf/mia_hand.xml`, Klasse `mia_ee_orientation`, von `±1.570796` auf `±1` reduziert (Joint-`range` bewusst unverändert gelassen – nur der kommandierbare Bereich wird begrenzt, nicht die physikalische Gelenkgrenze). Ergebnis: **0/32 Fehler** im vollen Stresstest-Set.

### Verifikation & Verfeinerung anhand echter Trainingsläufe

Ein 200k-Schritte-Verifikationslauf über alle 5 Algorithmen zeigte 4/5 vollständig erfolgreich (PPO, SAC, TD3, DDPG mit validen Metriken), **A2C** brach bei 91 % (182.500/200.000) über den Watchdog ab. Analyse des A2C-Trainingslogs kurz vor dem Abbruch: Policy-`std` (Explorationsstreuung) lag bei **1.34**, deutlich höher als bei den anderen Algorithmen zu vergleichbarem Zeitpunkt.

Gezielter Stresstest mit an A2C angelehntem Rauschprofil (`np.random.normal(0, 1.34, ...)` um einen zufälligen Bias) bei `limit=1.0`: **2/30 Fehler** (~7 %) – erklärt den beobachteten Ausreißer als seltenes, aber nicht verschwindendes Restrisiko bei hoher Explorationsstreuung. Nachjustierung auf `limit=0.9`: **0/160** über vier Aktionsmuster (je 40 Episoden). Fix aktualisiert auf `ctrlrange="-0.9 0.9"`. Ein anschließender realer 200k-Schritte-A2C-Lauf mit diesem Wert lief fehlerfrei durch (einzelne "Nan/Inf"-Warnungen traten weiterhin gelegentlich auf, führten aber nur noch zu regulärem Episodenabbruch statt zum Hängenbleiben des Prozesses).

### Nicht (mehr) beobachtete, aber dokumentierte Nebenwirkung

Ein echter, unabhängiger Bug wurde parallel gefunden und behoben: `MJRobot.set_ctrl`/`MiaHand.set_ctrl`/`MiaHand._set_mrl_ctrl` (`deformable_gym/robots/mj_robot.py`) akkumulieren Aktionen **additiv** auf den aktuellen Ctrl-Wert (`new_ctrl = data.ctrl[id] + ctrl[i]`), ohne Clipping auf `model.actuator_ctrlrange`. Bei sustained/biased Policies lief der Ctrl-Wert unbegrenzt von der physikalischen Aktuatorgrenze weg. Fix: neuer Helper `_clip_to_ctrlrange()`, an allen drei Stellen eingebaut, respektiert `model.actuator_ctrllimited`. Für sich genommen löste dieser Fix das Kernproblem **nicht** (identische Fehlerzeitpunkte davor/danach im Stresstest), ist aber unabhängig davon korrekt und bleibt Teil des Fixes (verhindert, dass Aktuator-Ziele über ihre physikalische Reichweite hinauswandern).

## 6. Reporting-Tool (`analyze_results.py`, neu)

Aggregiert alle `env/algorithm/seed`-Läufe unter `results/` (erkannt über `grasp_stability.csv`). Erzeugt:
- `summary_table.csv`/`.md`: eine Zeile pro (env, algorithm), über Seeds gemittelt (± Std bei >1 Seed).
- `plots/bar_*.png`: Balkenvergleich der Algorithmen je Metrik.
- `plots/curve_*.png`: Trainingsverlauf je Metrik über Timesteps.
- `report.html`: eigenständige Seite (Tabelle + alle Plots), `color-scheme: light` erzwungen für konsistente Darstellung unabhängig vom Browser-Dark-Mode.

**Robustheits-Detail:** Wenn ein Run vom Watchdog abgebrochen wurde, enthält `summary.csv` nur eine `timed_out=True`-Platzhalterzeile ohne Metriken. `load_summary()` erkennt das (`mean_reward` fehlt) und fällt für diese Zeile auf die letzte Zeile der laufeigenen `grasp_stability.csv` zurück – so gehen bei einem Teilabbruch keine bereits erreichten Zwischenergebnisse verloren. Tabelle enthält eine `Status`-Spalte (`vollstaendig` / `abgebrochen (Zeitlimit)`) sowie `Trainingsschritte erreicht`, um Vergleiche nicht ohne Kontext über unterschiedlich weit trainierte Policies zu präsentieren.

## 7. Validierungsergebnisse (finaler 1M-Schritte-Lauf, `MjFloatingMiaGraspBoxes-v0`)

| Algorithmus | Trainingsschritte erreicht | Status | Instabilitätsanteil (Eval) | Aktionssättigung | Reward (Test) |
|---|---|---|---|---|---|
| SAC | 1.000.000 | vollständig | **0 %** | 0.131 | −1 |
| A2C | 1.000.000 | vollständig | **0 %** | 1.0 | −1 |
| DDPG | 1.000.000 | vollständig | **0 %** | 1.0 | −1 |
| PPO | 520.000 | abgebrochen (Watchdog) | 99 % (kurz vor Abbruch) | 0.999 | −1 |
| TD3 | 1.000.000 | vollständig (lief durch) | **100 %** | 1.0 | 0 |

**Interpretation:**
- Für SAC/A2C/DDPG ist das ursprüngliche Instabilitätsproblem auf dieser Skala vollständig verschwunden.
- PPO zeigt ein selteneres Restrisiko (vorher Abbruch nach ~14.000 Schritten, jetzt nach 520.000 – über 35× länger stabil, aber noch nicht bei 0).
- TD3 zeigt einen **qualitativ anderen** Fehlermodus: kein Hänger, aber die konvergierte Policy sättigt praktisch immer ihre Aktionen und produziert dadurch bei jeder Eval-Episode Instabilität – ein Policy-Kollaps-Phänomen, kein Simulator-Bug. Interessant im Kontext von Aktionssättigung als Frühindikator (auch A2C/DDPG zeigen 100 % Sättigung, aber ohne Instabilität – Sättigung allein ist also nicht hinreichend, vermutlich spielt *welche* Aktuatoren/Richtung eine Rolle).
- Kein Algorithmus hat bei 1 Mio. Schritten Sparse-Reward einen erfolgreichen Griff gelernt (erwartbar, war nicht Ziel dieses Laufs).

## 8. Geänderte/neue Dateien (Überblick)

- `deformable_gym/helpers/mj_utils.py` — `get_body_subtree_ids`, `get_direct_child_body_names`
- `deformable_gym/helpers/grasp_metrics.py` — neu
- `deformable_gym/envs/mujoco/base_mjenv.py` — Energie-Flag, gecachte Body-IDs
- `deformable_gym/envs/mujoco/grasp_env.py` — Info-Dict-Integration
- `deformable_gym/robots/mj_robot.py` — `_clip_to_ctrlrange`
- `deformable_gym/assets/robots/mjcf/mia_hand.xml` — `ctrlrange` der ee-Orientierungsaktuatoren `±1.570796 → ±0.9`
- `pipeline.py` — komplette Neufassung (Watchdog, Callbacks, CLI)
- `analyze_results.py` — neu
- `train_test.py` — gelöscht (funktional ersetzt durch `pipeline.py`)

## 9. Offene Punkte / Future Work

1. **PPO-Restfall (520k):** noch nicht auf denselben Wert wie A2C nachgeschärft/verifiziert; vermutlich hilft eine weitere leichte Reduktion der `ctrlrange` oder eine gesonderte Untersuchung des spezifischen DOF/Zeitpunkts.
2. **TD3-Politik-Kollaps:** eher ein RL-/Explorationsproblem als ein Simulationsproblem – ggf. Action-Noise-Konfiguration von TD3 (Default `NormalActionNoise`) oder ein Sättigungs-Penalty im Reward untersuchen.
3. Kein Reward-Shaping mit den neuen Grasp-Measures – aktuell rein Beobachtungsgrößen, bewusst nicht in den Reward eingemischt (siehe Abschnitt 3); für spätere Arbeiten (z. B. Contact Score als dichtes Zusatzsignal) vorbereitet, aber nicht aktiviert.
4. `energy_quality` ist eine grobe Näherung (mischt Gravitations-/Elastizitätsenergie) – für eine sauberere Trennung wäre ein plugin-spezifischer Zugriff auf die Elastizitäts-Energie nötig (in MuJoCo aktuell nicht direkt exponiert).
