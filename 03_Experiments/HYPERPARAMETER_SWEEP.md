# Iteration 6: PPO/SAC/A2C-Hyperparameter-Sweep

**Projektübersicht:** Siehe [[../../README]]

**Ausgangslage:** Fünf Ablationsiterationen (200k–2 Mio. Schritte, sparse/Phase-1/1+2/1+2+3, mit und ohne Warmstart-Curriculum, s. [[REWARD_SHAPING_ABLATION]]) haben nie eine Policy hervorgebracht, die echtes, durch Training erworbenes Greifen zeigt. DDPG/TD3 zeigen dabei einen eigenständigen, bereits identifizierten Fehlermodus (Policy-Kollaps: eingefrorene, vollständig gesättigte Policy ab dem ersten Checkpoint). PPO/SAC/A2C kollabieren nicht, aber sie konvergieren auch nicht auf Kontakt – sie bleiben über alle bisherigen Iterationen hinweg flach bei `n_contacts≈0`. Alle bisherigen Läufe verwendeten durchgängig SB3-Standard-Hyperparameter (`pipeline.py`: `algo_class("MlpPolicy", train_env, verbose=0, seed=seed)`, keine `learning_rate`/`ent_coef`/`policy_kwargs`-Anpassung).

**Teil dieser Session:** Phase 8 (Hyperparameter-Sweeps, Iteration 6). Siehe [[../02_Sessions/SESSION_CONTEXT_AND_FINDINGS]] für den kompletten Kontext.

**Verwandte Dokumentation (Dokumentations-Struktur):**

- [[../02_Sessions/SESSION_NOTES_grasp_stability_pipeline]] – Phase 1: Grasp-Stability-Measures & Trainings-Pipeline-Entwicklung
- [[../02_Sessions/SESSION_CONTEXT_AND_FINDINGS]] – Gesamtkontext aller Iterationen (Phase 2–8) und Empfehlungen
- [[REWARD_SHAPING_ABLATION]] – Phases 7–8: Reward-Shaping-Iterationen 1–5 mit Ablationsergebnissen

**Ziel dieser Iteration:** Statt einer erneuten Breitband-Ablation (Reward-Profile × Algorithmen) wird für die drei nicht-kollabierenden Algorithmen (PPO, SAC, A2C) gezielt an den SB3-Hyperparametern gedreht, auf dem bislang vielversprechendsten Setup (`phase1_warmstart`-Profil, s. Iteration 5) fixiert.

## Umsetzung

- `pipeline.py`: neuer optionaler `algo_kwargs`-Parameter auf `run_single`/`run_with_watchdog`, durchgereicht an den SB3-Algorithmus-Konstruktor (`algo_class("MlpPolicy", train_env, verbose=0, seed=seed, **(algo_kwargs or {}))`). Default `None` reproduziert das alte Verhalten bit-identisch.
- `hparam_configs.py` (neu): Single Source of Truth für die drei Varianten je Algorithmus (siehe Tabelle unten).
- `run_hparam_sweep.py` (neu): orchestriert den Sweep, analog zu `run_reward_ablation.py`, aber mit fixem Reward-Profil und variierenden `algo_kwargs` statt variierenden `env_kwargs`. Ergebnisse unter `results/hparam_sweep/<algo>/<variant>/seed<seed>/`.

## Varianten (je Algorithmus genau eine Achse gegenüber dem SB3-Default verändert)

| Variante | PPO | SAC | A2C | Begründung |
|---|---|---|---|---|
| Default (Referenz) | `net_arch=[64,64]`, `ent_coef=0.0`, `lr=3e-4` | `net_arch=[256,256]`, `ent_coef="auto"`, `lr=3e-4` | `net_arch=[64,64]`, `ent_coef=0.0`, `lr=7e-4` | Bereits aus Iteration 5 (`results/ablation/phase1_warmstart/`) vorhanden, hier nicht erneut gelaufen |
| `larger_net` | `net_arch=[256,256]` | `net_arch=[400,400]` | `net_arch=[256,256]` | SAC ist per Default schon bei [256,256] – Variante geht dort bewusst weiter auf [400,400], um noch eine echte Vergrößerung zu sein |
| `high_entropy` | `ent_coef=0.02` | `ent_coef=0.1` (überschreibt `"auto"`) | `ent_coef=0.02` | PPO/A2C haben ohne diese Änderung **keinerlei** Entropie-Bonus (Default 0.0) |
| `high_lr` | `lr=1e-3` | `lr=1e-3` | `lr=2e-3` | ~3× Default, algorithmusspezifisch skaliert |

**Umfang:** 3 Algorithmen × 3 Varianten = 9 Läufe, 400.000 Schritte, `phase1_warmstart`-Profil, 1 Seed (gleiches Zeitbudget-Argument wie in den Voriterationen).

```bash
python run_hparam_sweep.py --algorithms PPO SAC A2C --profile phase1_warmstart --total-timesteps 400000 --eval-freq 20000
```

Gestartet am 2026-07-10 im Hintergrund (Log: `ablation_v6_hparam_sweep.log`), abgeschlossen am 2026-07-11 (alle 9 Läufe durchgelaufen bzw. per Watchdog beendet).

## Ergebnisse

### Laufstatus

7 von 9 Läufen (78 %) per Watchdog beendet – höher als die ~50–53 % aus den Iterationen 3–5. Nur `SAC/larger_net` und `A2C/high_lr` liefen vollständig durch; alle 3 PPO-Varianten sowie `SAC/high_entropy`, `SAC/high_lr`, `A2C/larger_net`, `A2C/high_entropy` wurden abgebrochen.

### Best-Checkpoint-Übersicht (alle 9 Läufe)

| Algo | Variante | `best_n_contacts_mean` | `best_grasped_mean` | `best_step` | `best_action_saturation_mean` | Status |
|---|---|---|---|---|---|---|
| PPO | larger_net | 0.0 | 0.0 | 40000 | 0.951 | Timeout |
| PPO | high_entropy | – | – | – | – | Timeout (kein Checkpoint erreicht) |
| PPO | high_lr | 0.0 | 0.0 | 100000 | 0.962 | Timeout |
| **SAC** | **larger_net** | **7.0** | **1.0** | 60000 | **0.012** | **vollständig** |
| SAC | high_entropy | 0.0 | 0.0 | 40000 | 0.005 | Timeout |
| SAC | high_lr | 0.0 | 0.0 | 20000 | 0.055 | Timeout |
| A2C | larger_net | 0.0 | 0.0 | 80000 | 1.0 | Timeout |
| A2C | high_entropy | 0.0 | 0.0 | 60000 | 0.992 | Timeout |
| A2C | high_lr | 0.0 | 0.0 | 380000 | 1.0 | vollständig |

### Kernbefund: `SAC/larger_net` zeigt erstmals nicht-kollabiertes Verhalten mit wiederholtem echtem Kontakt

Alle anderen 8 Läufe reproduzieren bekannte Muster: entweder durchgehend `n_contacts=0` (PPO in allen 3 Varianten, SAC/high_entropy, SAC/high_lr) oder das seit Iteration 3/5 dokumentierte Policy-Kollaps-Muster (`A2C/high_lr`: `action_saturation=1.0` bereits ab dem ersten Checkpoint bei Schritt 20.000, Reward ab Schritt 300.000 bit-identisch eingefroren bei 3.5299988858460254, `n_contacts=0` durchgehend – kein Lernfortschritt).

**`SAC/larger_net` (Netzwerkgröße 256→400 Neuronen pro Schicht) unterscheidet sich davon qualitativ:** `grasp_stability.csv` zeigt echten Kontakt an **zwei unabhängigen, weit auseinanderliegenden Checkpoints** – Schritt 60.000 (`n_contacts=7`, Reward 7.02) und Schritt 320.000 (`n_contacts=16`, Reward 7.02) – mit normal schwankendem, nicht-gesättigtem Verhalten dazwischen (`action_saturation` zwischen 1 % und 42 %, Reward variiert Checkpoint zu Checkpoint zwischen −1.5 und 7.0, keine bit-identischen Wiederholungen). Das ist das erste Mal in sechs Iterationen, dass eine Policy **wiederholt und unabhängig voneinander** echten Kontakt herstellt, ohne dabei in eine eingefrorene, gesättigte Dauerlösung zu kollabieren – auch wenn sie den Griff noch nicht über die gesamte restliche Trainingszeit stabil beibehält.

### Empfehlung / nächster Schritt

`SAC/larger_net` ist der einzige Befund dieser Iteration, der eine gezielte Vertiefung rechtfertigt, statt der Breite nach mehr Varianten zu testen:

1. **Mehrere Seeds** (Reproduzierbarkeit prüfen – bislang nur Seed 0) für `SAC/larger_net`, gleiches Setup (400k Schritte, `phase1_warmstart`).
2. **Längeres Training** für einen Seed, um zu prüfen, ob der wiederholt gefundene Kontakt sich mit mehr Zeit zu einem stabilen Verhalten festigt statt nur wiederkehrend aufzutreten.

Beides gestartet als Iteration 6b, siehe unten.

## Iteration 6b: `SAC/larger_net`-Folgelauf (Multi-Seed + verlängertes Training)

**Ziel:** Prüfen, ob der in Iteration 6 beobachtete wiederholte (aber nicht dauerhafte) echte Kontakt bei `SAC/larger_net` reproduzierbar ist (mehrere Seeds) und ob er sich bei längerem Training zu einem stabilen Griff festigt.

**Umfang:**
- 4 zusätzliche Seeds (1–4) bei unverändertem Budget (400.000 Schritte, `phase1_warmstart`, gleiche `algo_kwargs`) – Seed 0 liegt bereits unter `results/hparam_sweep/SAC/larger_net/seed0/` vor und wird nicht erneut gelaufen.
- 1 verlängerter Lauf (Seed 0, 1.200.000 Schritte statt 400.000 – 3× Budget) in einem separaten Ergebnisordner, um eine Überschreibung des bereits vorliegenden 400k-Ergebnisses zu vermeiden.

```bash
# Multi-Seed (Reproduzierbarkeit)
python run_hparam_sweep.py --algorithms SAC --variants larger_net --seeds 1 2 3 4 --total-timesteps 400000 --eval-freq 20000 --results-dir ./results/hparam_sweep

# Verlängertes Training (Seed 0, separates Verzeichnis)
python run_hparam_sweep.py --algorithms SAC --variants larger_net --seeds 0 --total-timesteps 1200000 --eval-freq 40000 --results-dir ./results/hparam_sweep_sac_larger_net_long
```

Gestartet am 2026-07-11 im Hintergrund (Logs: `ablation_v6b_sac_multiseed.log`, `ablation_v6b_sac_long.log`), abgeschlossen am 2026-07-18.

### Ergebnis: Seed 0 war ein Einzelfall, nicht reproduzierbar

**Multi-Seed (1–4):** Keiner der vier neuen Seeds zeigt echten Kontakt – `best_n_contacts_mean=0.0` und `best_grasped_mean=0.0` bei allen vieren, durchgängig. Der in Iteration 6 bei Seed 0 beobachtete wiederholte, nicht-kollabierte Kontakt (Schritt 60k und 320k) **reproduziert sich nicht**. Timeout-Quote 2/4 (50 %, Seeds 2 und 4), passend zum sonstigen Bild dieser Studie.

**Verlängertes Training (Seed 0, 1,2 Mio. Schritte geplant):** Lief per Watchdog nur bis Schritt ~703.000 (58 % des geplanten Budgets, `run-timeout-minutes=240` erreicht, kein Absturz/Hänger). In den zusätzlichen ~300.000 Schritten über das ursprüngliche 400k-Budget hinaus (Checkpoints 440k–680k) taucht **kein neuer Kontakt-Erfolg** auf – der bereits aus Iteration 6 bekannte beste Checkpoint bleibt Schritt 320.000 (`n_contacts=16`). Positiv: Reward und `action_saturation` schwanken über den gesamten verlängerten Zeitraum weiterhin normal (0.06–0.46) statt einzufrieren – die Policy zeigt also nicht das DDPG/TD3/A2C-typische Kollaps-Muster, findet aber auch keinen zusätzlichen, dauerhaften Griff.

**Einordnung:** Der Seed-0-Befund aus Iteration 6 war rückblickend am ehesten ein seedspezifischer Zufallstreffer (zwei vereinzelte Kontakt-Episoden während der Exploration), keine reproduzierbare Eigenschaft von `SAC/larger_net`. Damit bleibt über alle 6 Iterationen und jetzt auch 6b weiterhin **kein einziger reproduzierbarer, echter Grasp-Erfolg** bestehen.

### Empfehlung

`SAC/larger_net` als eigenständiger Hebel ist damit erledigt – kein weiterer Folgelauf gerechtfertigt. Vielversprechender ist die parallel gelaufene Iteration 7 (DDPG/TD3-Action-Noise-Diagnose, siehe unten): dort zeigt zumindest TD3 mit Rauschen ein qualitativ neues, nicht-kollabiertes Verhalten, das einen gezielten Folgelauf rechtfertigt.

## Iteration 7: DDPG/TD3-Action-Noise-Diagnose

**Ausgangslage:** Der DDPG/TD3-Policy-Kollaps (eingefrorene, vollständig gesättigte Policy ab dem ersten Eval-Checkpoint, s. [[../02_Sessions/SESSION_CONTEXT_AND_FINDINGS]] Abschnitt 9) wurde in Iteration 5/6 zurückgestellt zugunsten des PPO/SAC/A2C-Sweeps. Recherche in den SB3-Defaults ergab einen konkreten, bislang übersehenen Verdächtigen: `DDPG.__init__`/`TD3.__init__` setzen `action_noise=None` als Default. Beide Algorithmen haben **deterministische** Policies – anders als PPO/SAC/A2C gibt es also gar kein eingebautes Explorationsrauschen während des Trainings, sofern man es nicht explizit konfiguriert. Kein einziger Lauf in dieser Studie (Iterationen 1–6) hat das je getan.

**Ziel:** Diese Hypothese schnell und günstig testen (60.000 statt 400.000 Schritte, ~10–20 Minuten statt Stunden pro Lauf), bevor eine vollskalige Ablation gestartet wird.

**Umsetzung:** Neues Skript `diagnose_action_noise.py`, das für DDPG und TD3 je einen `no_noise`-Lauf (unveränderte Defaults, reproduziert das bekannte Kollaps-Muster) und einen `with_noise`-Lauf (`NormalActionNoise`, Std-Abweichung 20 % der jeweiligen Aktionsdimension) auf dem `phase1_warmstart`-Profil durchführt, unter Wiederverwendung von `pipeline.run_with_watchdog`/`algo_kwargs` (s. Iteration 6).

```bash
python diagnose_action_noise.py --algorithms DDPG TD3 --total-timesteps 60000 --eval-freq 10000 --run-timeout-minutes 60
```

Gestartet und abgeschlossen am 2026-07-12 (Log: `diagnose_action_noise.log`).

### Ergebnis: Hypothese nur für TD3 bestätigt, für DDPG widerlegt

| Lauf | Verhalten über 6 Checkpoints (10k–60k) |
|---|---|
| DDPG/no_noise | Bit-identisch ab dem ersten Checkpoint (`n_contacts=11`, Reward 7.02, `action_saturation=1.0` durchgehend) – reproduziert exakt das aus Iteration 5 bekannte Muster. |
| DDPG/with_noise | **Ebenfalls eingefroren** – ab Checkpoint 2 bit-identisch (Reward ≈ −0.301, `n_contacts=0`, `action_saturation=1.0` durchgehend), nur an einem anderen (schlechteren) Fixpunkt als ohne Rauschen. |
| TD3/no_noise | Checkpoints 1–3 (10k–30k) zeigen `n_contacts=3`, Reward 7.02 – dann harter Übergang bei Schritt 40k zu einem neuen, ab da eingefrorenen Zustand (Reward 0.09, `n_contacts=0`) bis Schritt 60k. Reproduziert das aus Iteration 6 bekannte "nur beim ersten Checkpoint"-Muster. |
| **TD3/with_noise** | **Kein Einfrieren:** Reward variiert normal über alle 6 Checkpoints (1.37 → 1.37 → −0.54 → −1.35 → 1.13 → 0.12), `action_saturation` bewegt sich (1.0 → 0.996 → 1.0 → 0.94 → 0.49 → 0.58) statt bei 1.0 zu verharren. **Aber:** `n_contacts=0` bei allen 6 Checkpoints – noch kein echter Kontakt gefunden, vermutlich weil 60k Schritte für "aus dem eingefrorenen Zustand heraus tatsächlich lernen" zu kurz sind. |

Wichtige technische Randbemerkung für die Interpretation: `action_noise` wirkt bei SB3 nur während der **Trainings-Rollouts** (Explorationsverhalten, das den Replay-Buffer befüllt), nicht beim deterministischen Eval (`GraspMetricsEvalCallback` ruft immer `model.predict(obs, deterministic=True)` auf). Der Unterschied zwischen `no_noise` und `with_noise` zeigt sich hier also indirekt – über eine unterschiedlich trainierte, aber weiterhin deterministisch ausgewertete Policy –, nicht direkt als Rauschen in den geloggten Metriken selbst.

**Interpretation:** Für TD3 erklärt fehlendes Explorationsrauschen einen Teil des Kollaps-Musters – mit Rauschen bleibt die Policy in Bewegung statt an einem Fixpunkt zu verharren, auch wenn sie in dieser kurzen Laufzeit noch keinen Kontakt herstellt. Für DDPG reicht dieselbe Erklärung nicht aus: Die Policy kollabiert auch mit Rauschen, nur an einer anderen Stelle – hier muss die Ursache woanders liegen (z. B. Critic-Divergenz, `learning_starts`, Actor-Netzwerk-Initialisierung).

### Empfehlung für Iteration 8

1. **TD3 mit `action_noise` in einem vollen Lauf testen** (400.000 Schritte, `phase1_warmstart`), um zu prüfen, ob sich das nicht-eingefrorene Verhalten aus dieser Kurzdiagnose bei mehr Trainingszeit zu echtem, gehaltenem Kontakt entwickelt.
2. **DDPG separat untersuchen** – `action_noise` allein reicht hier nicht; als Nächstes Actor-Netzwerk-Initialisierung, `learning_starts` und Critic-Lernrate gezielt variieren, analog zur Methodik dieser Iteration (kurze 60k-Diagnoseläufe vor einer vollen Ablation).

## Iteration 8: TD3 mit `action_noise` im vollen Lauf

**Ziel:** Punkt 1 der Iteration-7-Empfehlung umsetzen – prüfen, ob das in der 60k-Kurzdiagnose beobachtete nicht-eingefrorene Verhalten von TD3 mit `NormalActionNoise` bei vollem Trainingsbudget zu echtem, gehaltenem Kontakt führt statt nur die Policy in Bewegung zu halten. Punkt 2 (DDPG-spezifische Untersuchung von Actor-Init/`learning_starts`/Critic-Lernrate) ist bewusst nicht Teil dieser Iteration – eigenständiges Setup, kein einfaches Hochskalieren von `diagnose_action_noise.py`.

**Umsetzung:** Wiederverwendung von `diagnose_action_noise.py` (Iteration 7) unverändert, nur mit vollem Budget statt der 60k-Kurzdiagnose. Läuft `TD3/no_noise` (Referenz, reproduziert das bekannte Kollaps-Muster) und `TD3/with_noise` (`NormalActionNoise`, σ = 20 % der Aktionsspannweite) sequentiell auf dem `phase1_warmstart`-Profil.

```bash
python diagnose_action_noise.py --algorithms TD3 --total-timesteps 400000 --eval-freq 20000 --run-timeout-minutes 240 --results-dir ./results/diagnose_action_noise_iter8_td3
```

Gestartet am 2026-07-18 im Hintergrund (Log: `ablation_v8_td3_action_noise.log`), abgeschlossen am 2026-07-18.

### Ergebnis: Rauschen verhindert das Einfrieren, aber der Lauf hängt sich stattdessen auf

**`TD3/no_noise`** (vollständig durchgelaufen, alle 400.000 Schritte): reproduziert exakt das aus Iteration 5/7 bekannte Muster. `n_contacts=3` bei Schritt 20.000 (einziger Checkpoint mit Kontakt), danach **bit-identisch eingefroren** bei Reward 0.0906, `n_contacts=0`, `action_saturation=1.0` über alle 19 folgenden Checkpoints (40k–400k). Bestätigt den Kollaps als robust reproduzierbar auch bei vollem Budget.

**`TD3/with_noise`**: kollabiert nicht, findet aber auch keinen Kontakt – und der Lauf bricht vorzeitig ab.

| Schritt | Reward | `action_saturation` | `n_contacts` |
|---|---|---|---|
| 20.000 | 1.37 | 0.996 | 0.0 |
| 40.000 | −1.35 | 0.942 | 0.0 |
| 60.000 | 0.12 | 0.584 | 0.0 |
| 80.000 | −0.36 | 0.531 | 0.0 |

Reward und `action_saturation` bewegen sich sichtbar über alle 4 erreichten Checkpoints (`action_saturation` sinkt sogar kontinuierlich von 99,6 % auf 53 %) – das Rauschen verhindert also tatsächlich das Kollaps-Muster, wie in der Iteration-7-Kurzdiagnose vermutet. **Aber:** Der Lauf wurde vom Watchdog nach nur ~81.700 von 400.000 Schritten gekillt (`no progress for 15 min` – ein Hänger, kein regulärer Truncate durch Instabilität). Das ist der aus der allerersten Session-Phase bekannte MuJoCo-`mj_step`-Hänger (ein einzelner Physik-Schritt kehrt bei extremen Zuständen nie zurück, s. `SESSION_NOTES_grasp_stability_pipeline.md`) – vermutlich provoziert durch das zusätzliche Aktionsrauschen, das die Simulation gelegentlich in einen Extremzustand drängt, den die `mia_hand`-`ctrlrange`-Fixes aus Phase 1 nicht abdecken. In den 4 erreichten Checkpoints: durchgehend `n_contacts=0`.

**Einordnung:** `action_noise` löst das Policy-Kollaps-Problem zuverlässig, tauscht es aber gegen einen neuen, härteren Fehlermodus (Simulations-Hänger statt sauberem Truncate) ein. Ob es bei vollständigen 400k Schritten zu echtem Kontakt käme, bleibt unbeantwortet, weil der Lauf technisch nie so weit kam.

### Empfehlung für Iteration 9

1. **Hänger-Ursache mit `action_noise` gezielt untersuchen**, bevor ein erneuter voller Lauf sinnvoll ist – z. B. kleineres `--noise-sigma-fraction` (aktuell 20 % der Aktionsspannweite, evtl. zu aggressiv in Kombination mit der bekannten Instabilitäts-Anfälligkeit dieser Simulation) oder ein Noise-Decay-Schedule statt konstantem Rauschen über die ganze Laufzeit.
2. Falls das den Hänger behebt: erneuter voller 400k-Lauf – aus der Iteration-6b-Lehre **von Anfang an mit mehreren Seeds**, nicht erst nachträglich, um einen Einzel-Checkpoint-Zufallstreffer nicht wieder als Durchbruch fehlzudeuten. Ebenfalls aus Iteration 6b: `n_eval_episodes` erhöhen (aktuell 3).
3. Alternativ: DDPG-spezifische Untersuchung (Actor-Init, `learning_starts`, Critic-Lernrate) aus der ursprünglichen Iteration-7-Empfehlung Punkt 2, die bislang noch nicht angegangen wurde.

## Iteration 9: DDPG-spezifische Kollaps-Untersuchung

**Ziel:** Punkt 3 der Iteration-8-Empfehlung umsetzen. Iteration 7 hat gezeigt, dass `action_noise` bei DDPG (anders als bei TD3) den Kollaps **nicht** verhindert – die Policy friert trotzdem ein, nur an einem anderen Fixpunkt. Der Fehlermodus muss also woanders liegen. Kandidaten: Netzwerkkapazität, `learning_starts` (Default 100 – nach nur 100 Zufallsschritten übernimmt bereits der Actor die Exploration), Lernrate.

**Wichtige Einschränkung:** SB3s `DDPG.__init__` hat **eine gemeinsame** `learning_rate` für Actor und Critic (kein separater `critic_learning_rate`-Kwarg) und exponiert keine direkte Gewichts-Initialisierung – "Actor-Netzwerk-Initialisierung" wird hier daher über Netzwerkkapazität (`policy_kwargs.net_arch`) angenähert, nicht über ein echtes Init-Schema.

**Umsetzung:** Neues Skript `diagnose_ddpg.py`, analog zu `diagnose_action_noise.py` (Iteration 7): 4 Varianten, je eine Achse gegenüber dem SB3-Default verändert, alle in derselben kurzen Diagnose-Skala (60k Schritte, `eval_freq=10000`) direkt vergleichbar:

| Variante | Änderung | Begründung |
|---|---|---|
| `baseline` | keine (`learning_rate=1e-3`, `learning_starts=100`, `net_arch=[400,300]`) | Referenz, in diesem Lauf direkt mitgeführt statt nur auf frühere Läufe zu verweisen |
| `high_learning_starts` | `learning_starts=10000` (100×) | Mehr Zufalls-Explorationsdaten im Replay-Buffer, bevor überhaupt ein Gradientenschritt läuft |
| `low_lr` | `learning_rate=1e-4` (10× kleiner) | Langsamerer Actor-Drift, falls die Standard-Rate ihn zu schnell in die Tanh-Sättigung treibt |
| `larger_net` | `policy_kwargs.net_arch=[512,512]` | DDPG ist per Default bereits bei `[400,300]` (anders als PPO/A2C) – Variante geht bewusst darüber hinaus |

```bash
python diagnose_ddpg.py --total-timesteps 60000 --eval-freq 10000 --run-timeout-minutes 60 --results-dir ./results/diagnose_ddpg_iter9
```

Gestartet am 2026-07-18 im Hintergrund (Log: `ablation_v9_ddpg_diagnostic.log`). Ergebnisse folgen in einer Aktualisierung dieses Abschnitts.
