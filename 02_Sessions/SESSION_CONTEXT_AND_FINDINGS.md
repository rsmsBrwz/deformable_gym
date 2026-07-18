# Deformable-Grasp-RL (Boxes/mia_hand): Gesamt-Session-Kontext & Erkenntnisse

**Zweck dieses Dokuments:** (1) Kontext-Datei für die Fortsetzung dieser Arbeit in einer neuen Session, (2) Rohnotizen für schriftliche Ausarbeitungen zum Thema. Fasst eine mehrtägige Arbeitssession (2026-07-03 bis 2026-07-11) zusammen: von "Environment testen" über eine tiefe Simulationsinstabilitäts-Diagnose bis zu sechs Iterationen einer Reward-Shaping-/Curriculum-/Hyperparameter-Ablationsstudie.

**Verwandte Dokumentation (Dokumentations-Struktur):**

- [[SESSION_NOTES_grasp_stability_pipeline]] – Phase 1: Grasp-Stability-Measures & Trainings-Pipeline-Entwicklung (2026-07-03 bis 2026-07-05)
- [[../03_Experiments/REWARD_SHAPING_ABLATION]] – Phases 7–8: Reward-Shaping-Iterationen 1–5 mit Ablationsergebnissen
- [[../03_Experiments/HYPERPARAMETER_SWEEP]] – Phase 8: SB3-Hyperparameter-Sweeps (Iteration 6 & 6b)

**Repo:** `deformable_gym`, Branch `development`. **Provenienz:** Der ursprüngliche Commit `d5d5fad` referenzierte in seiner Nachricht mehrere neue Dateien (`grasp_metrics.py`, `shaped_grasp_env.py`, `boxes.xml`, `reward_profiles.py`, `run_reward_ablation.py`, `analyze_results.py`, alle drei Doku-Dateien), die tatsächlich als `??` (untracked) im Repo lagen statt committed zu sein – eine Lücke aus einem vorherigen `git add`, die erst am 2026-07-08 per `git commit --amend` (kein Remote-Tracking auf diesem Branch, daher unproblematisch) geschlossen wurde; aktueller Hash dafür: `ad663f4`. Iteration 5 (Warmstart-Curriculum, s.u.) ist als eigener Commit `81868c3` sauber getrennt. Damit ist der Hinweis aus früheren Versionen dieses Dokuments ("noch nicht committed") überholt.

---

## 1. Ausgangslage und Auftrag

Ziel zu Beginn: Ein neues, ungetestetes MuJoCo-Environment (`MjFloatingMiaGraspBoxes-v0`, 3 deformierbare Boxen, `mia_hand`-Roboter) mit mehreren Stable-Baselines3-Algorithmen (PPO/SAC/TD3/A2C/DDPG) trainieren, dabei Grasp-Stability-Measures erfassen und Trainingskennzahlen für spätere Analyse speichern – mit Blick auf begrenzte lokale Rechenzeit. Später erweitert um: Ergebnis-Reporting für eine Doktoranden-Präsentation, dann um systematisches Reward Shaping (nach einer vom User bereitgestellten Konzeptvorlage) mit einer begleitenden, reproduzierbaren Ablationsstudie.

## 2. Projektarchitektur (Kurzreferenz)

- `BaseMJEnv` → `GraspEnv` (`deformable_gym/envs/mujoco/`): Szene = Robot-MJCF + Objekt-MJCF + `mj_scene_base.xml`, gemergt via `deformable_gym/helpers/asset_manager.py:create_scene`.
- Reward (Original, **unverändert** in `GraspEnv`): sparse, 0 während der Episode, am Ende (`_pause_simulation`: löst Objekt-Fixierung, friert Robotergelenke ein, 1s warten) `+1` wenn Objekthöhe > 0.2, sonst `-1`.
- Objekt `boxes`: 3 unabhängige `flexcomp`-Soft-Bodies (`box_left/center/right`), je 8 Vertex-Bodies mit je 3 Slide-Joints (kein Rotations-DOF pro Vertex).
- Roboter `mia_hand` bei `control_type="joint"`: Policy steuert **sowohl** Finger-Aktuatoren **als auch** die 6 Basis-Pose-Aktuatoren (`ee_A_X/Y/Z/OX/OY/OZ`) – nicht offensichtlich, war Ursache eines frühen Analysefehlers (siehe 4.5).
- Registrierung aller Kombinationen automatisch in `deformable_gym/__init__.py` (`register_mj_grasp_envs`).

## 3. Chronologie / Phasen dieser Session

1. **Projektanalyse** – Architektur verstanden, festgestellt: `pipeline.py`/`train_test.py` (alt) waren funktional kaputt, keine Grasp-Stability-Measures vorhanden.
2. **Grasp-Stability-Measures implementiert** (neues Modul `grasp_metrics.py`) + saubere `pipeline.py`-Neufassung mit SB3-Monitor/Logger/Eval-Callback.
3. **Smoke-Test → großer Trainingslauf → 11,5h-Hang-Incident** – Entdeckung, dass ein einzelner `mj_step`-Aufruf nach Instabilität nie zurückkehren kann.
4. **Watchdog-Architektur** (Subprozess + Heartbeat) gegen genau dieses Hang-Problem.
5. **Root-Cause-Diagnose der Instabilität** – Ausschlussverfahren über 5 falsche Hypothesen, am Ende gefunden: kinematische Singularität (Gimbal-Lock) in der Handorientierung.
6. **Ergebnis-Reporting-Tool** (`analyze_results.py`) für Präsentationszwecke.
7. **Reward-Shaping-Architektur** (`ShapedGraspEnv`, neue Datei, `GraspEnv` bewusst nicht verändert) nach externer Konzeptvorlage.
8. **Vier Ablations-Iterationen** (sparse vs. Phase 1/1+2/1+2+3-Reward), dabei ein weiterer echter Bug gefunden und behoben (fehlende Weld-Constraints in `boxes.xml`) und Best-Checkpoint-Tracking ergänzt.

---

## 4. Die zentrale Untersuchung: Simulationsinstabilität – Hypothesen-Chronologie

**Symptom:** Bei realistischer Trainingsskala (200k–1M+ Schritte) destabilisiert sich die MuJoCo-Simulation reproduzierbar (`Nan, Inf or huge value in QACC`-Warnung); einmal hing der gesamte Prozess **11,5 Stunden** ohne jeden Fortschritt (0% CPU-Fortschritt sichtbar in Logs, aber 98% CPU-Auslastung – ein einzelner `mj_step`-Aufruf kehrte nie zurück, vermutlich weil eine nicht-finite Position die Broad-Phase-Kollisionserkennung kombinatorisch explodieren ließ).

**Methodik:** Ein Stresstest-Script (feste Aktionsmuster: `constant_strong`, `constant_moderate`, `biased_noise`, `random_uniform`) gegen `MjFloatingMiaGraspBoxes-v0`, Instabilität erkannt über (a) `np.isfinite(obs)` und (b) `data.time` monoton steigend (ein zweites, unabhängig entdecktes Symptom: nach der Instabilitätswarnung lief `data.time` teils **nicht-monoton**). Baseline: 15/32 Fehler (47%), praktisch 100% bei `constant_strong`.

### Tabelle: Getestete Hypothesen

| # | Hypothese | Testmethode | Ergebnis | Verdikt |
|---|---|---|---|---|
| 1 | Elastizitätsmodell der Boxen (`young`/`damping` in `boxes.xml`) zu weich/steif | `damping` 1e-4→1e-2 (100×) | Bit-identische Fehlerzeitpunkte | **Verworfen** |
| 2 | Integrator (explizit vs. implizit) | `model.opt.integrator` → `mjINT_IMPLICITFAST` zur Laufzeit | Kein Effekt | **Verworfen** |
| 3 | `<edge equality>`-Constraint der Flexcomp zu steif | Constraint entfernt | Kein Effekt | **Verworfen** |
| 4 | Kontakt-Solver zu steif | `mjENBL_OVERRIDE` + diverse `o_solref`/`o_solimp`/`o_margin` | Kaum Effekt (13-14/16 statt 15/16) | **Verworfen** |
| 5 | Aktuatorkraft zu hoch | `model.actuator_forcerange` auf 50%/25%/10% skaliert | Kein Effekt (8/8 Fehler bei jeder Skalierung) | **Verworfen** |
| 6 | Ctrl-Wert akkumuliert unbegrenzt über physikalische Aktuatorgrenze hinaus (`MJRobot.set_ctrl`: `new_ctrl = data.ctrl[id] + ctrl[i]` ohne Clipping) | Fix implementiert (`_clip_to_ctrlrange`), Stresstest wiederholt | **Realer Bug**, aber identische Fehlerzeitpunkte davor/danach im Stresstest | **Bestätigt als Bug, aber nicht die Hauptursache** – Fix trotzdem beibehalten (korrekt, unabhängig relevant) |
| 7 | **Kinematische Singularität (Gimbal-Lock)** in den 3 sequentiellen Hinge-Joints der Handorientierung (`ee_A_OX/OY/OZ`, `ctrlrange` bis exakt ±π/2) | `model.actuator_ctrlrange` systematisch von 1.5708 auf 1.0 verkleinert, Sweep 1.1–1.5708 | Scharfer Übergang zwischen 1.4 (8/8 fail) und 1.3 (0/8) | **Bestätigt – Hauptursache.** Nebenbefund: `shadow_hand.xml` begrenzt dieselben Aktuatoren bereits auf ±1 rad, deshalb dort nie aufgefallen |
| 7b | Residualfehler bei A2C trotz Fix (11% Zeitlimit-Abbruch bei 91% Fortschritt) | A2C hatte auffällig hohe Policy-`std` (1.34); Stresstest mit `std=1.34`-Rauschen bei `limit=1.0`: 2/30 Fehler | Bestätigt Restrisiko bei hoher Explorationsstreuung | `ctrlrange` weiter auf **±0.9** verschärft (0/160 über 4 Muster) |

**Analysefehler, transparent dokumentiert:** Ein Zwischentest ("Basis-Aktuatoren auf 0 setzen, nur Finger testen") beruhte auf einer falschen Annahme über die `robot.actuators`-Reihenfolge (`[thumb, index, ee_X, ee_Y, ee_Z, ee_OX, ee_OY, ee_OZ, ...]`, nicht wie angenommen `[ee_X...ee_OZ, thumb, index, ...]`). Der scheinbare Beleg für eine separate Ursache löste sich nach Korrektur auf.

### Zweiter Root-Cause-Fund (nach den Reward-Shaping-Iterationen 1-2): Boxen fielen unabhängig von der Policy

- **Hypothese (falsch):** Hand-Objekt-Abstand zu groß, Policy driftet ab, bevor Kontakt entsteht.
- **Test:** `model.body(...).pos` nach Reset geprüft → Hand und Box bereits **bei Reset in Kontakt** (4 Kontaktpunkte, `grasped=True`). Abstand war nie das Problem.
- **Tatsächlicher Befund:** Bei Null-Aktion über die ganze Episode fällt die Box innerhalb von **20 Schritten (~0,3s)** von Höhe 0.495 auf Bodenhöhe 0.017 – **unabhängig von jeder Policy-Aktion**. Vergleich mit `insole_fixed`/`pillow_fixed` unter identischem Test: Höhe bleibt dort konstant.
- **Ursache gefunden:** `insole_fixed.xml` hat einen expliziten `<equality><weld name="fix_*" body1="..." body2="insole_fixed" /></equality>`-Block (8 Welds), der das Objekt gegen die Schwerkraft verankert, bis `_pause_simulation()` am Episodenende alle Equality-Constraints löst. `boxes.xml` hatte nur `<edge equality="true">` (hält nur die *eigene* Form starr, keine Verankerung an die Welt) – **ein echter Konfigurationsfehler**, keine RL-/Reward-Frage.
- **Fix:** 24 Welds (alle 8 Vertex-Bodies je der 3 Boxen) analog zu `insole_fixed.xml` ergänzt. Kein Code musste geändert werden (`_pause_simulation()` löst über `eq_constraints_to_disable` automatisch auch die neuen Constraints). Verifiziert: Box-Höhe bleibt jetzt über die volle Episode konstant.
- **Wichtige Erkenntnis danach:** Dieser Fix war **notwendig, aber nicht hinreichend** – siehe Ablations-Iterationen 2–4 unten, das Kontaktproblem blieb trotzdem bestehen.

---

## 5. Implementierte Komponenten (chronologisch, alle unversioniert / uncommitted)

| Datei | Was |
|---|---|
| `deformable_gym/helpers/mj_utils.py` | + `get_body_subtree_ids`, `get_direct_child_body_names` |
| `deformable_gym/helpers/grasp_metrics.py` (neu) | `ContactScore`, `hand_object_contact_ids`, `contact_score`, `binary_grasp_state`, `object_retained_ratio`, `energy_quality`, `dynamic_stability_probe` (inkl. `settle_step`) |
| `deformable_gym/envs/mujoco/base_mjenv.py` | Energie-Flag (`mjENBL_ENERGY`), gecachte Hand-/Objekt-Body-IDs; **neu (Iteration 5):** `warm_start_joint_targets`-Kwarg + `_apply_warm_start()`, setzt qpos **und** das treibende Aktuator-`ctrl` beim Reset, Default `None` verhält sich bit-identisch zum alten Code |
| `deformable_gym/envs/mujoco/grasp_env.py` | Info-Dict-Integration der Grasp-Measures (episodische Keys immer NaN-vorbelegt, damit SB3 `Monitor(info_keywords=...)` nie crasht); reicht `warm_start_joint_targets` an `BaseMJEnv` durch |
| `deformable_gym/robots/mj_robot.py` | `_clip_to_ctrlrange()`-Helper (Bug #6 oben) |
| `deformable_gym/assets/robots/mjcf/mia_hand.xml` | ee-Orientierung `ctrlrange` `±1.5708 → ±0.9` (Hauptfix #7) |
| `deformable_gym/assets/objects/mjcf/boxes.xml` | 24 Weld-Constraints ergänzt (Fix #2, s.o.) |
| `deformable_gym/envs/mujoco/shaped_grasp_env.py` (neu) | `ShapedGraspEnv(GraspEnv)` – 4-stufige Reward-Architektur, s. Abschnitt 6; **Iteration 5:** Phase-3-Rewards (Retention/Energie/Dynamic) jetzt auf `retained_ratio > 0` gegatet |
| `deformable_gym/__init__.py` | `register_mj_grasp_envs_shaped()` – neue Env-IDs `Mj*Grasp*Shaped-v0`, rein additiv |
| `pipeline.py` (komplette Neufassung + mehrere Erweiterungen) | Watchdog-Subprozess-Architektur, `NanSafeWrapper`, `GraspMetricsEvalCallback` (inkl. Best-Checkpoint-Tracking), `env_kwargs`-Durchreichung |
| `reward_profiles.py` (neu) | 4 benannte Reward-Profile: `sparse`, `phase1`, `phase1_2`, `phase1_2_3` |
| `run_reward_ablation.py` (neu) | Ablations-Orchestrierung über `reward_profiles.py` × Algorithmen × Seeds |
| `analyze_results.py` (neu) | Tabellen-/Plot-/HTML-Report aus `results/<env>/<algo>/seed<seed>/` |
| `train_test.py` | **gelöscht** (funktional durch `pipeline.py` ersetzt) |
| `SESSION_NOTES_grasp_stability_pipeline.md` | Technische Doku Phasen 1–6 |
| `REWARD_SHAPING_ABLATION.md` | Technische Doku der 4 Ablations-Iterationen (Phasen 7–8) |

---

## 6. Reward-Shaping-Architektur (`ShapedGraspEnv`)

**Wichtige Vorgabe (User):** `GraspEnv` darf nicht verändert werden – `ShapedGraspEnv` ist eine reine Subklasse in neuer Datei, registriert unter eigenen Env-IDs (z.B. `MjFloatingMiaGraspBoxesShaped-v0`).

4 Reward-Komponenten, alle einzeln gewichtbar (Konstruktor-Kwargs, Defaults siehe Tabelle), Reihenfolge in `_compute_total_reward` **wichtig** (Kommentare im Code):
1. `_compute_stability_step_reward()` – per-step `grasped`/Kontaktqualität, **muss vor** dem Task-Reward laufen (der via `_pause_simulation` den Zustand verändert).
2. `_compute_event_reward()` – Drop-Penalty, Acquire-Bonus, Progress-Reward (Vergleich zum Vorschritt).
3. `_compute_task_reward()` – ruft unverändert `GraspEnv._get_reward()`.
4. `_compute_terminal_stability_reward()` – liest `self._episode_grasp_metrics` (von `_pause_simulation` befüllt): Retention/Energie/Dynamic-Stability.

Zusätzlicher Fix nur in dieser Subklasse: Terminierungs-Timing-Bug in `GraspEnv.step()` (`sim_time` wird **vor** `mj_step` erfasst, nicht danach) – in `ShapedGraspEnv.step()` korrigiert, in `GraspEnv` bewusst unverändert gelassen.

| Parameter | Default | Phase |
|---|---|---|
| `w_task` | 1.0 | Basis (Task-Erfolg, unverändert) |
| `w_grasped`, `w_contact` | 0.01, 0.01 | 1 (aktiv) |
| `w_drop` | -0.5 | 1 (aktiv) |
| `w_acquire`, `w_progress` | 0.0 (Ablation: 0.5, 0.2) | 2 (opt-in) |
| `w_retained`, `w_energy`, `w_dynamic` | 0.0 (Ablation: 0.2, 0.1, 0.1) | 3 (opt-in) |
| Normierung | `contact_force_saturation=5N`, `energy_reference=1.0`, `dynamic_displacement_reference=0.01` | grobe Erstkalibrierung, noch nicht empirisch verfeinert |

Alle 13 `reward_*`-Unterkomponenten werden im Info-Dict separat geloggt (`reward_total`, `reward_task_terminal`, `reward_stability_step`, `reward_event`, `reward_terminal_stability`, `reward_grasped`, `reward_contact`, `reward_drop`, `reward_acquire`, `reward_progress`, `reward_retention`, `reward_energy`, `reward_dynamic`).

## 7. Pipeline-Erweiterungen für die Ablationsstudie (`pipeline.py`)

- `env_kwargs`-Parameter durchgereicht `make_env()` → `run_single()` → `run_with_watchdog()` (rückwärtskompatibel, Default `None`).
- `GraspMetricsEvalCallback.extra_info_keys` – zusätzliche, tolerant geloggte Keys (NaN falls nicht vorhanden, kein Crash) – genutzt für die 13 `reward_*`-Keys, die es nur bei `ShapedGraspEnv` gibt (nicht bei sparse `GraspEnv`).
- **Best-Checkpoint-Tracking** (letzte Ergänzung): Score = `mean_reward - 2.0 * sim_unstable_mean` (Instabilität dominiert die Bewertung, damit ein durch Truncation "zufällig gut aussehender" Reward nicht fälschlich als bester Checkpoint zählt). Speichert `checkpoints/best_model.zip` + `checkpoints/best_checkpoint.json` (Score, Formel, volle Metriken). `summary.csv`/`grasp_stability.csv` bekommen `best_step` + alle `best_*`-Spalten. Auch bei Watchdog-Abbruch wird der beste bis dahin gefundene Checkpoint nachträglich aus der Datei übernommen.
- Watchdog-Architektur (aus Phase 4): jede (env, algo, seed)-Kombination läuft in eigenem `multiprocessing.Process` (`spawn`), `HeartbeatCallback` meldet alle 200 Steps Fortschritt über eine `Queue`. Zwei Limits: `--idle-timeout-minutes` (Default 15, Lücke zwischen Heartbeats) und `--run-timeout-minutes` (Default 240, absolute Obergrenze). Grund: ein einzelner hängender `mj_step`-Aufruf ist nur per `SIGKILL` von außen stoppbar, kein Python-Timeout reicht.

## 8. Ablationsstudien: 9 Iterationen im Überblick

Iterationen 1–5 über `run_reward_ablation.py` (Reward-Profile × Algorithmen), Ergebnisse gesichert unter `results/ablation_*`/`results/ablation/`. Siehe [[../03_Experiments/REWARD_SHAPING_ABLATION]] für Iterationen 1–2. Iterationen 6–9 über `run_hparam_sweep.py`/`diagnose_action_noise.py`/`diagnose_ddpg.py` (SB3-Hyperparameter bzw. `action_noise`/DDPG-spezifische Parameter statt Reward-Profile variiert), Ergebnisse unter `results/hparam_sweep*/`/`results/diagnose_action_noise*/`/`results/diagnose_ddpg*/`. Siehe [[../03_Experiments/HYPERPARAMETER_SWEEP]] für Details zu Iterationen 6–9.

| Iteration | Umfang | Kernbefund |
|---|---|---|
| 1 (200k, 4 Profile × 5 Algo) | `results/ablation_prefix/` | Mechanik korrekt (Gewichte/Logging funktionieren), aber **kein Lauf** erreicht je Kontakt (`n_contacts=0` durchgängig) → Phase 1/2 nie ausgelöst. Phase 3 liefert kontaktunabhängigen Sockelbonus (~+0.05, `dynamic_stability_probe` bewertet auch ein ruhendes, nie gegriffenes Objekt positiv). |
| Zwischenschritt | – | Root-Cause gefunden: Boxen fallen (s. Abschnitt 4) mangels Weld-Constraint. Fix umgesetzt. |
| 2 (200k, nach Weld-Fix) | `results/ablation_200k_iter2/` | Fix technisch korrekt (Box fällt nicht mehr), **aber Kontakt bleibt weiterhin bei 0** in allen abgeschlossenen Läufen. Erste Erkenntnis: Fix war nötig, aber nicht hinreichend. |
| 3 (2 Mio., reduziert auf `sparse`+`phase1_2_3`) | `results/ablation_2m_iter3/` | **Wichtigster Zwischenbefund:** `sparse/A2C` zeigte bei 900k–1.2M Schritten `n_contacts=1, grasped=True, reward=0` – aber gleichzeitig `sim_unstable_mean=1.0`. Verdacht: Instabilitäts-Artefakt, kein echter Erfolg. DDPG/TD3 zeigten Politik-Kollaps (dauerhaft instabil ab 100k) bzw. komplettes Einfrieren (identischer Reward über alle 2 Mio. Schritte). **50% Timeout-Quote**, höher als bei kürzeren Läufen. |
| 4 (400k, alle 4 Profile, mit Best-Checkpoint-Tracking) | `results/ablation_400k_iter4/` | **Bestätigt:** Score-basierte Best-Checkpoint-Auswahl (die Instabilität herausrechnet) zeigt über **alle 20 Läufe** `best_n_contacts_mean=0.0` – der A2C-Befund aus Iteration 3 war tatsächlich ein Instabilitäts-Artefakt. Über 4 Iterationen (>40 Läufe) **kein einziger instabilitätsbereinigter echter Grasp-Erfolg**. Erneut ~50% Timeout-Quote, gleichmäßig über alle Profile (auch `sparse`) verteilt. |
| 5 (400k, 3 neue Warmstart-Profile × 5 Algo) | `results/ablation/` | Umsetzung des naheliegendsten offenen Hebels aus Iteration 4 (Curriculum/Warmstart): Hand startet direkt über der linken Box statt in Zufallsdistanz. Kalibrierung per Zero-Action-Rollout zeigt `grasped`-Anteil 99.8 % (vs. 27 % Standardpose) – die Kontakt-Vorbedingung ist damit technisch hergestellt. **Trotzdem kein echter, durch Training gelernter Grasp-Erfolg:** `sparse_warmstart` bleibt bei 0/5 wie in allen Voriterationen; der einzige scheinbare Erfolg (`phase1_warmstart`/DDPG, `n_contacts=11` über alle 20 Checkpoints identisch) ist ein von Anfang an eingefrorenes, vollständig gesättigtes Policy-Kollaps-Artefakt (`action_saturation=1.0` durchgehend), kein Lernfortschritt. 53 % Timeout-Quote, unverändert hoch. |
| 6 (400k, PPO/SAC/A2C-Hyperparameter-Sweep, 3 Varianten je Algo) | `results/hparam_sweep/` | Statt weiterer Reward-Profile: SB3-Hyperparameter variiert (Netzwerkgröße, Entropie-Koeffizient, Lernrate) auf dem `phase1_warmstart`-Profil, für die 3 nicht-kollabierenden Algorithmen (DDPG/TD3 ausgeklammert, s. Empfehlung aus Iteration 5). **`SAC/larger_net` (Netzwerk 256→400) zeigt als erster Lauf über 6 Iterationen wiederholten, nicht-kollabierten echten Kontakt** – `n_contacts>0` an zwei unabhängigen, weit auseinanderliegenden Checkpoints (Schritt 60k und 320k), mit normal schwankendem statt eingefrorenem Verhalten dazwischen. Alle anderen 8 Läufe reproduzieren bekannte Muster (`n_contacts=0` durchgängig oder A2C/high_lr-Policy-Kollaps). Timeout-Quote 78 % (7/9), höher als je zuvor. **Iteration 6b (Folgelauf) widerlegt den Befund:** 0/4 neue Seeds zeigen Kontakt, verlängertes Training (bis Watchdog-Abbruch bei ~700k von geplant 1,2 Mio. Schritten) bringt keinen neuen Erfolg – Seed 0 war ein Einzelfall, s. [[../03_Experiments/HYPERPARAMETER_SWEEP]]. |
| 7 (60k-Kurzdiagnose, DDPG/TD3 mit/ohne `action_noise`) | `results/diagnose_action_noise/` | Getestet, ob SB3s `action_noise=None`-Default (deterministische Policies ohne jedes Explorationsrauschen) den DDPG/TD3-Kollaps erklärt. **Nur für TD3 bestätigt:** mit Rauschen bleibt die Policy über alle 6 Checkpoints in Bewegung statt einzufrieren, findet in der kurzen Laufzeit aber noch keinen Kontakt. **Für DDPG widerlegt:** friert trotz Rauschen weiterhin ein, nur an einem anderen (schlechteren) Fixpunkt. |
| 8 (400k, TD3 mit `action_noise` im vollen Lauf) | `results/diagnose_action_noise_iter8_td3/` | `TD3/no_noise` reproduziert bei vollem Budget exakt das bekannte Kollaps-Muster (Kontakt nur bei Schritt 20k, danach 19 Checkpoints bit-identisch eingefroren). `TD3/with_noise` kollabiert erwartungsgemäß nicht (Reward/`action_saturation` bleiben in Bewegung), **findet aber ebenfalls keinen Kontakt und bricht nach nur ~82k von 400k Schritten mit einem MuJoCo-`mj_step`-Hänger ab** (Watchdog: „no progress for 15 min“) – ein neuer, härterer Fehlermodus statt einer Lösung. Ob mehr Trainingszeit zu echtem Kontakt geführt hätte, bleibt offen. |
| 9 (60k-Kurzdiagnose, DDPG `learning_starts`/Lernrate/Netzgröße) | `results/diagnose_ddpg_iter9/` | `baseline` und `larger_net` frieren wie gewohnt früh ein (10k bzw. 20k); `high_learning_starts` verzögert den Kollaps nur um einen Checkpoint und landet dann auf einem schlechteren Fixpunkt. **`low_lr` (1e-3→1e-4) ist die erste DDPG-Konfiguration der gesamten Studie, die über alle 6 Checkpoints nicht einfriert** (Reward/`action_saturation` bleiben durchgehend in Bewegung) – findet aber ebenfalls keinen Kontakt, analog zu `TD3/with_noise` aus Iteration 8. |

### Widerlegte Hypothese: "Mehr Trainingsschritte lösen das Kontaktproblem"

200k → 2 Mio. Schritte (10× mehr) hat die Grundsituation nicht verbessert (Iteration 3), die Instabilitätsquote eher verschlechtert. **Klar widerlegt als alleiniger Hebel.**

### Widerlegte Hypothese: "Der Erfolg war da, wir haben nur falsch ausgewertet"

Best-Checkpoint-Tracking (Iteration 4) mit instabilitätsbereinigter Score-Funktion zeigt weiterhin 0 Erfolge. **Widerlegt** – das Problem liegt nicht an der Metrik-Auswertung.

## 9. Offene Fragen / Empfehlungen für die nächste Session

1. ~~**`SAC/larger_net`-Folgelauf**~~ – **abgeschlossen, Befund widerlegt (Iteration 6b, s. [[../03_Experiments/HYPERPARAMETER_SWEEP]]):** Weder die 4 neuen Seeds (1–4, alle `best_n_contacts_mean=0.0`) noch das verlängerte Training (Seed 0, bis Watchdog-Abbruch bei ~700k Schritten kein neuer Erfolg über den bekannten Checkpoint 320k hinaus) reproduzieren den in Iteration 6 beobachteten wiederholten Kontakt. War vermutlich ein seedspezifischer Zufallstreffer, keine robuste Eigenschaft von `SAC/larger_net`.
2. ~~**TD3 mit `action_noise` im vollen Lauf**~~ – **abgeschlossen, Ergebnis uneindeutig (Iteration 8, s. [[../03_Experiments/HYPERPARAMETER_SWEEP]]):** `TD3/no_noise` reproduziert bei 400k Schritten exakt das bekannte Kollaps-Muster (Kontakt nur bei Schritt 20k, danach 19× bit-identisch eingefroren). `TD3/with_noise` kollabiert zwar nicht (Reward/`action_saturation` bleiben in Bewegung), **hängt sich aber nach nur ~82k Schritten mit dem aus der allerersten Session-Phase bekannten MuJoCo-`mj_step`-Hänger auf** (Watchdog: „no progress for 15 min“), bevor überhaupt genug Zeit für Kontakt bliebe – kein Erfolg, aber auch kein sauber widerlegter Ansatz, da der Lauf technisch nie ans Ziel kam. Nächster Schritt (Iteration 9): kleineres `--noise-sigma-fraction` oder ein Decay-Schedule testen, um den Hänger zu vermeiden, bevor ein erneuter voller Lauf sinnvoll ist – diesmal von Anfang an mit mehreren Seeds und höherem `n_eval_episodes` (Lehren aus Iteration 6b).
3. ~~**DDPG-Policy-Kollaps – Kurzdiagnose**~~ – **abgeschlossen, ein Hoffnungsschimmer (Iteration 9, s. [[../03_Experiments/HYPERPARAMETER_SWEEP]]):** `learning_starts` und Netzgröße helfen nicht (Kollaps bleibt oder verschiebt sich nur auf einen schlechteren Fixpunkt). **`low_lr` (Lernrate 1e-3→1e-4) ist die erste DDPG-Konfiguration der gesamten Studie, die über 60k Schritte nicht einfriert** – findet aber noch keinen Kontakt. **Aktueller Stand:** `low_lr` läuft als Iteration 10a im Hintergrund auf vollem 400k-Budget weiter (mehrere Seeds von Anfang an, höheres `n_eval_episodes` – Lehren aus Iteration 6b/8), während parallel Iteration 10b (PPO/SAC/A2C breiter bzw. Environment-seitiger Hebel) aktiv bearbeitet wird.
4. **A2C zeigt jetzt ebenfalls ein Kollaps-Muster (neu in Iteration 6):** `A2C/high_lr` friert ab Schritt 300.000 bit-identisch ein (`action_saturation=1.0` bereits ab dem ersten Checkpoint) – bislang nur bei DDPG/TD3 beobachtet, jetzt auch bei einem on-policy-Algorithmus unter höherer Lernrate. Sollte bei der DDPG-Kollaps-Untersuchung (Punkt 3) mit betrachtet werden.
5. **~50–78%-Timeout-Quote als eigenständiges Problem, über 8 Iterationen nie unter 50%:** Iteration 6 zeigt mit 78 % die bislang höchste Quote; Iteration 8 fügt mit dem `mj_step`-Hänger unter `action_noise` einen weiteren, qualitativ anderen Instabilitäts-Fall hinzu. Tritt gleichmäßig über Profile/Varianten auf – mit mehreren Seeds prüfen, ob das reproduzierbar an bestimmten Algorithmus-Kombinationen hängt oder zufällig verteilt ist. Bislang nur mit **1 Seed** gearbeitet (Zeitbudget-Entscheidung, Ausnahme: der Iteration-6b-Folgelauf) – Konfidenz aller Ablationsbefunde entsprechend begrenzt.
6. **`retained_ratio > 0`-Gate (Iteration 5, s. [[../03_Experiments/REWARD_SHAPING_ABLATION]]) schließt die Phase-3-Sockelbonus-Lücke nur teilweise:** Sie kann bereits durch minimalen Kontakt beim Pause-Check am Episodenende feuern, auch wenn während der Episode selbst nie `grasped=True` gemessen wurde (beobachtet bei `phase1_2_3_warmstart`/DDPG: `retained_ratio_mean=0.333` bei `grasped_mean=0.0`). Für eine vollständige Schließung müsste die Gate zusätzlich einen Mindest-`grasped`/`n_contacts`-Wert *während* der Episode voraussetzen.
7. `energy_quality` bleibt eine grobe Näherung (mischt Gravitations-/Elastizitätsenergie) – für eine sauberere Trennung wäre ein plugin-spezifischer Zugriff auf die Elastizitäts-Energie nötig (in MuJoCo aktuell nicht direkt exponiert).
8. Die Warmstart-Zielpose (Iteration 5) ist auf **eine** Box (`box_left`) kalibriert, nicht auf alle drei symmetrisch – für eine spätere Verallgemeinerung (z. B. zufällige Box-Auswahl pro Episode) müsste die Pose relativ zur tatsächlichen Objektposition berechnet werden statt als fixer Offset.
9. `n_eval_episodes=3` (alle Iterationen bislang) macht Best-Checkpoint-Metriken anfällig für Einzel-Episoden-Rauschen – Ursache des Iteration-6-Fehlschlusses (s. Iteration 6b). Für zukünftige Läufe erhöhen.

## 10. Links und Navigation

**Projektüberblick:**
- [[../../README]] – Projekt-Übersicht und Installation
- [[../../CONTRIBUTING]] – Beitrittsrichtlinien

**Verwandte Dokumentationen (diese Session):**
- [[SESSION_NOTES_grasp_stability_pipeline]] – Anfangliche Grasp-Measures und Pipeline-Entwicklung
- [[../03_Experiments/REWARD_SHAPING_ABLATION]] – Reward-Shaping-Ablationen im Detail
- [[../03_Experiments/HYPERPARAMETER_SWEEP]] – Hyperparameter-Optimierungen

---

## 11. Schnellreferenz: Wie reproduziere ich was

```bash
# Einzelner Trainingslauf (Smoke-Test-Skala)
python pipeline.py --algorithms PPO --total-timesteps 3000 --eval-freq 1500

# Volle Ablationsstudie (alle 4 Profile × 5 Algorithmen)
python run_reward_ablation.py --total-timesteps 400000 --eval-freq 20000

# Reduzierte Ablation (nur 2 Profile, für schnelleren Vergleich)
python run_reward_ablation.py --profiles sparse phase1_2_3 --total-timesteps 2000000 --run-timeout-minutes 480

# Iteration 5: nur die Warmstart-Profile
python run_reward_ablation.py --profiles sparse_warmstart phase1_warmstart phase1_2_3_warmstart --total-timesteps 400000 --eval-freq 20000

# Reporting für einen beliebigen results-Ordner
python analyze_results.py --results-dir ./results/ablation
```

Alle Trainingsläufe laufen in `.venv-mj` (separates venv mit mujoco 3.2.3, gymnasium 1.2.2, stable-baselines3 2.7.1) – Aufruf entsprechend über `.venv-mj/bin/python ...`.
