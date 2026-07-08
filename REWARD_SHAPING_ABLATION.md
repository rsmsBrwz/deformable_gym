# Reward-Shaping-Ablationsstudie: sparse vs. Phase 1/1+2/1+2+3

**Datum:** 2026-07-05 (Iteration 1), aktualisiert 2026-07-06 (Iteration 2, nach Root-Cause-Fix)
**Git-Commit zum Zeitpunkt beider Läufe:** `da60daf93d8952b2684a332d37a7cfa975ac5580` (siehe Provenienz-Hinweis weiter unten — `boxes.xml` ist unversioniert, der Hash unterscheidet Iteration 1/2 daher nicht)
**Vollständige Maschinenkonfiguration:** [results/ablation/ablation_config.json](results/ablation/ablation_config.json) (Iteration 2, aktuell) bzw. [results/ablation_prefix/ablation_config.json](results/ablation_prefix/ablation_config.json) (Iteration 1, archiviert)

## Ziel

Empirische Erstkalibrierung der in [shaped_grasp_env.py](deformable_gym/envs/mujoco/shaped_grasp_env.py) implementierten Reward-Shaping-Architektur (siehe auch die vorherige Session-Doku [SESSION_NOTES_grasp_stability_pipeline.md](SESSION_NOTES_grasp_stability_pipeline.md)). Frage: Verhalten sich die vier Reward-Stufen (sparse Baseline, Phase 1, Phase 1+2, Phase 1+2+3) wie im Konzeptdokument (`reward_shaping_plan_grasp_env.md`) vorgesehen, und sind die gewählten Gewichte sinnvoll kalibriert?

## Methodik

### Reproduktion

```bash
python run_reward_ablation.py
# äquivalent mit expliziten Defaults:
python run_reward_ablation.py \
  --profiles sparse phase1 phase1_2 phase1_2_3 \
  --algorithms PPO SAC TD3 A2C DDPG \
  --seeds 0 --total-timesteps 200000 --eval-freq 20000
```

Danach:
```bash
python analyze_results.py --results-dir ./results/ablation
```

### Profile ([reward_profiles.py](reward_profiles.py) — single source of truth)

| Profil | Env | Aktive Gewichte |
|---|---|---|
| `sparse` | `MjFloatingMiaGraspBoxes-v0` (unverändertes `GraspEnv`) | — |
| `phase1` | `MjFloatingMiaGraspBoxesShaped-v0` | Defaults: `w_task=1.0, w_grasped=0.01, w_contact=0.01, w_drop=-0.5` |
| `phase1_2` | `MjFloatingMiaGraspBoxesShaped-v0` | + `w_acquire=0.5, w_progress=0.2` |
| `phase1_2_3` | `MjFloatingMiaGraspBoxesShaped-v0` | + `w_retained=0.2, w_energy=0.1, w_dynamic=0.1` |

Normierungskonstanten (unverändert seit Implementierung, siehe vorherige Session-Notiz): `contact_force_saturation=5.0N`, `energy_reference=1.0`, `dynamic_displacement_reference=0.01`.

### Umfang

Alle 5 SB3-Algorithmen (PPO, SAC, TD3, A2C, DDPG) × 4 Profile × 1 Seed × 200.000 Schritte = 20 Läufe, über `pipeline.run_with_watchdog()` (gleiche Subprozess-/Heartbeat-/Timeout-Absicherung wie die regulären Trainingsläufe – ein hängender/instabiler Lauf wird nach max. 15 min Stillstand automatisch beendet und die Studie läuft mit der nächsten Kombination weiter).

### Technische Ergänzungen für diese Studie (rückwärtskompatibel, siehe `pipeline.py`)

- `env_kwargs`-Parameter durchgereicht durch `make_env()`/`run_single()`/`run_with_watchdog()`, um pro Lauf unterschiedliche Reward-Gewichte an `gym.make()` zu übergeben.
- `GraspMetricsEvalCallback` trackt jetzt zusätzlich optionale `extra_info_keys` (hier: die 13 `reward_*`-Komponenten aus `ShapedGraspEnv`). Für das `sparse`-Profil (hat diese Keys nicht) werden sie als `NaN` geloggt statt einen Fehler zu werfen – verifiziert in einem Dry-Run vor dem vollen Lauf.
- Ergebnisstruktur `results/ablation/<profil>/<algorithmus>/seed<seed>/` – das Profil nimmt bewusst die Position ein, die `analyze_results.py` bereits als "env" gruppiert; das bestehende Analyse-Tool wurde dafür **nicht verändert**.

## Ergebnisse

Vollständige Tabelle: [results/ablation/report/summary_table.csv](results/ablation/report/summary_table.csv) / `.md`, Plots: `results/ablation/report/plots/`, HTML-Report: [results/ablation/report/report.html](results/ablation/report/report.html).

### Laufstatus

3 von 20 Läufen wurden vom Watchdog beendet (Zeitlimit-Abbruch, kein Crash): `phase1/TD3`, `phase1_2/PPO`, `phase1_2_3/A2C`. Kein erkennbares systematisches Muster (unterschiedliche Algorithmen/Profile betroffen) – konsistent mit der bereits dokumentierten Restinstabilität (~5–10 % je Lauf), nicht erkennbar durch das Reward-Shaping verursacht.

### Kernbefund 1: Phase 1/2-Komponenten wurden in dieser Studie nie ausgelöst

In **keinem** der 17 abgeschlossenen Läufe kam es in der finalen Evaluation zu Handkontakt mit dem Objekt (`n_contacts_mean=0`, `grasped_mean=0` durchgängig). Entsprechend sind `reward_stability_step_mean` und `reward_event_mean` in **allen** Profilen exakt `0.0`:

| Profil | `reward_task_terminal` | `reward_stability_step` | `reward_event` | `reward_terminal_stability` | `reward_total` |
|---|---|---|---|---|---|
| phase1 | −1.0 | 0.0 | 0.0 | 0.0 | −1.0 |
| phase1_2 | −1.0 | 0.0 | 0.0 | 0.0 | −1.0 |
| phase1_2_3 | −1.0 | 0.0 | 0.0 | **+0.0496** | −0.928 bis −0.950 |

Bei 200.000 Schritten Sparse-Reward-Exploration lernt keiner der 5 Algorithmen, die Hand überhaupt in Objektnähe zu bewegen (`success_rate=0.0` über alle 20 Läufe). Damit ist diese Studie eine **korrekte Verifikation der Mechanik** (Gewichte kommen an, werden korrekt geloggt, brechen nichts), aber **keine** Aussage über die tatsächliche Lernwirkung von Phase 1/2 – die dafür nötige Kontakt-Erfahrung fand nie statt.

**Empfehlung für Iteration 2:** Entweder deutlich mehr Trainingsschritte (die Kontakt-Erfahrung muss erst zufällig durch Exploration entdeckt werden) oder eine Trainingskonfiguration mit häufigerem initialen Kontakt (z. B. engerer Start-Abstand Hand–Objekt, oder ein Curriculum), bevor die Phase-1/2-Gewichte selbst sinnvoll kalibriert werden können.

### Kernbefund 2: Phase 3 liefert einen kontaktunabhängigen Sockelbetrag

`reward_terminal_stability` (~+0.05) tritt in **jedem** terminierten `phase1_2_3`-Lauf nahezu identisch auf (0.0496–0.0497), unabhängig vom Algorithmus. Ursache: `dynamic_stability_probe` bewertet, wie schnell sich das Objekt nach einem Störimpuls beruhigt – das gilt trivial auch für ein nie gegriffenes, ruhendes Objekt. Der Bonus wird damit **unabhängig vom eigentlichen Greifverhalten** vergeben.

**Risiko (siehe auch "Risiko 3" im ursprünglichen Konzeptdokument):** Diese Komponente könnte in einer echten Trainingsphase geringfügig Reward-Hacking begünstigen (Agent lernt nichts über Greifen, bekommt aber einen kleinen, verlässlichen Bonus für Nichtstun). Angesichts der geringen Gewichtsgröße (`w_dynamic=0.1`, maximal ~0.1 von einem Task-Reward-Betrag von ±1.0) ist der Effekt aktuell klein, aber real.

**Empfehlung für Iteration 2:** `reward_dynamic`/`reward_retention` optional mit `retained_ratio > 0` oder dem Task-Erfolg gaten (z. B. nur auszahlen, wenn zumindest zeitweise Kontakt bestand), statt sie bedingungslos bei jeder Terminierung zu vergeben.

### Kernbefund 3: TD3-Instabilität ist algorithmus-, nicht profilspezifisch

`sim_unstable_mean=1.0` (100 % instabile Eval-Episoden) tritt bei TD3 in `sparse` und `phase1_2` auf (bei `phase1` timeout, daher kein Vergleichswert); außerdem bei A2C/`phase1_2` und DDPG/`phase1`. Das deckt sich mit dem bereits in der vorherigen Session dokumentierten TD3-"Policy-Kollaps"-Befund (siehe `SESSION_NOTES_grasp_stability_pipeline.md`, Abschnitt 7) – die Instabilität ist ein Optimierungs-/Explorationsphänomen der jeweiligen Algorithmen, nicht (primär) durch das Reward-Shaping verursacht oder behoben.

### Sonstige Beobachtungen

- `action_saturation_mean` variiert stark zwischen Algorithmen (SAC durchgängig niedriger, 0.25–0.59; PPO/A2C/DDPG/TD3 meist >0.95) unabhängig vom Profil – ein Algorithmus-, kein Shaping-Effekt.
- Kein Profil erreicht in dieser Studie `success_rate > 0`; ein Vergleich "verbessert Shaping die Erfolgsrate" ist mit diesem Datensatz noch nicht möglich.

## Fazit für die aktuelle Kalibrierung (Iteration 1)

Die Reward-Shaping-Implementierung ist **mechanisch korrekt und produktionsreif** (Gewichte, Logging, Pipeline-Integration alle verifiziert). Für eine **inhaltliche** Kalibrierung der Phase-1/2-Gewichte (`w_grasped, w_contact, w_drop, w_acquire, w_progress`) reicht diese Studie nicht aus, weil die Voraussetzung (tatsächlicher Objektkontakt während der Evaluation) nicht eintrat. Phase-3-Gewichte (`w_retained, w_energy, w_dynamic`) sollten vor einer größeren Studie um eine Kontakt-Bedingung ergänzt werden, um einen kontaktunabhängigen Sockelbonus zu vermeiden.

Ursachenanalyse direkt im Anschluss (siehe unten) ergab: Der fehlende Objektkontakt lag nicht am Reward-Shaping oder am Aktionsbereich, sondern an einem **Konfigurationsfehler in `boxes.xml`** — die Boxen hatten keinerlei Verankerung gegen die Schwerkraft und fielen unabhängig von der Policy binnen ~0,3 s zu Boden (siehe nächster Abschnitt).

## Zwischenschritt: Root-Cause-Fix (Weld-Constraints in `boxes.xml`)

Diagnose (Null-Aktion über die volle Episode, `MjFloatingMiaGraspBoxes-v0`):

| Schritt | Box-Höhe (COM z) |
|---|---|
| 0 | 0.495 |
| 20 | 0.017 (= Boden) |

Zum Vergleich `MjFloatingMiaGraspInsole-v0`/`...Pillow-v0` unter identischem Test: Höhe bleibt über 20 Schritte konstant (~0.45–0.48). Ursache gefunden: `insole_fixed.xml` besitzt einen expliziten `<equality><weld .../></equality>`-Block, der mehrere Netzknoten fest mit dem statischen Wurzel-Body verschweißt und das Objekt so gegen die Schwerkraft hält, bis `_pause_simulation()` am Episodenende alle Equality-Constraints löst. `boxes.xml` hatte nur `<edge equality="true">` (hält je Box nur die *eigene* Form starr, keine Verankerung an die Welt).

**Fix:** `<equality>`-Block mit 24 Welds (alle 8 Vertex-Bodies je der 3 Boxen, verschweißt mit ihrem jeweiligen Wurzel-Body `box_left`/`box_center`/`box_right`) ergänzt, exakt nach dem `insole_fixed.xml`-Muster. Kein Code musste geändert werden – `_pause_simulation()` löst über `eq_constraints_to_disable` automatisch auch diese neuen Constraints am Episodenende.

**Verifikation:** Box-Höhe bleibt jetzt über die volle Episode (400+ Schritte) bei Null-Aktion konstant (0.4956 durchgängig), fällt danach beim Constraint-Release korrekt auf Bodenhöhe – identisches Verhalten zu Insole/Pillow. Volle Test-Suite weiterhin grün (98 passed).

**Hinweis zur Provenienz:** `boxes.xml` ist wie schon zuvor eine unversionierte (nicht committete) Asset-Datei; der Git-Commit-Hash in `ablation_config.json` (`da60daf9...`) ändert sich dadurch nicht zwischen Iteration 1 und 2. Für exakte Reproduzierbarkeit über diesen Punkt hinaus sollte `boxes.xml` vor einer weiteren Iteration eingecheckt werden.

## Iteration 2: Ablation nach dem Weld-Fix

Gleicher Aufbau, gleiche 4 Profile, gleiche 5 Algorithmen, 1 Seed, 200.000 Schritte, erneut über `run_reward_ablation.py` (Ergebnisse in `results/ablation/`; die Iteration-1-Ergebnisse liegen zum Vergleich unverändert in `results/ablation_prefix/`).

### Laufstatus

Diesmal 6 von 20 Läufen per Watchdog beendet (vorher 3): `sparse/SAC`, `phase1/PPO`, `phase1/SAC`, `phase1/TD3`, `phase1_2/PPO`, `phase1_2_3/SAC`. Mit nur einem Seed lässt sich nicht sicher unterscheiden, ob das reines Sample-Rauschen ist oder ob das gleichzeitige Lösen von 24 statt vorher 0 Welds beim Episodenende einen etwas größeren numerischen Schock erzeugt (im Log dieser Iteration trat vereinzelt ein neuer Warnungstyp auf, `QVEL`-Ausreißer an einem Box-Vertex-DOF statt wie zuvor an einem Hand-DOF) – eine Beobachtung, kein bestätigter Befund. Bei einer Folgeiteration mit mehreren Seeds sollte das mitverfolgt werden.

### Kernbefund: Der Weld-Fix behebt das Kontakt-Problem nicht vollständig

Trotz korrekt verankerter Box bleibt `n_contacts_mean=0.0` / `grasped_mean=0.0` in **allen** 14 abgeschlossenen Läufen dieser Iteration weiterhin bei null – identisch zu Iteration 1. `reward_stability_step_mean` und `reward_event_mean` bleiben entsprechend weiterhin durchgängig `0.0`; `reward_terminal_stability` bleibt beim erwarteten kontaktunabhängigen Sockelbetrag (~0.050, phase1_2_3).

**Einordnung:** Der Weld-Fix war notwendig (er behebt einen echten Physik-Konfigurationsfehler und gibt der Policy jetzt die volle ~6s-Episode statt eines ~0,3s-Fensters), aber **nicht hinreichend**, um innerhalb von 200.000 Schritten Sparse-/Shaping-Exploration zu tatsächlichem Kontakt zu führen. Das Greifen selbst bleibt ein hartes Explorationsproblem, das durch die Objekt-Physik allein nicht gelöst wird.

**Empfehlung nach Iteration 2:** Da "mehr Zeit zum Fallen verhindern" jetzt erledigt ist, war der nächste Test, ob schlicht mehr Trainingsschritte das Kontakt-Problem lösen (siehe Iteration 3 unten).

## Iteration 3: 2 Mio. Schritte (sparse + phase1_2_3, alle 5 Algorithmen)

Reduzierter Umfang gegenüber der vollen 4-Profile-Ablation (Kosten/Zeit-Abwägung mit dem User): nur `sparse` und `phase1_2_3` (reinste Baseline vs. volles Shaping), alle 5 Algorithmen, 2.000.000 Schritte, `eval_freq=100000` (20 Auswertungspunkte je Lauf). `--run-timeout-minutes` von 240 auf 480 erhöht, da bei dieser Skala legitime (nicht hängende) Läufe für SAC/TD3/DDPG realistisch 3–4h dauern können. Ergebnisse in `results/ablation/`; die 200k-Iteration-2-Daten liegen zum Vergleich unverändert in `results/ablation_200k_iter2/`.

```bash
python run_reward_ablation.py --profiles sparse phase1_2_3 --total-timesteps 2000000 --eval-freq 100000 --run-timeout-minutes 480
```

### Laufstatus

5 von 10 Läufen (50 %!) wurden vom Watchdog beendet: `sparse/PPO`, `sparse/SAC`, `phase1_2_3/PPO`, `phase1_2_3/SAC`, `phase1_2_3/A2C`. Das ist eine deutlich höhere Abbruchquote als bei 200k Schritten (Iteration 2: 30 %) oder 1M Schritten in der früheren Session (siehe `SESSION_NOTES_grasp_stability_pipeline.md`). Mit nur einem Seed lässt sich nicht zweifelsfrei zwischen "längeres Training erhöht das kumulative Risiko, irgendwann die Restinstabilität zu treffen" und Stichproben-Rauschen unterscheiden, aber die Richtung ist bemerkenswert: **länger trainieren scheint das Instabilitätsrisiko nicht zu senken, tendenziell eher zu erhöhen.**

### Kernbefund: Mehr Trainingsschritte lösen das Kontakt-Problem nicht zuverlässig

Von den 5 abgeschlossenen Läufen zeigt **keiner** am Ende (2.000.000 Schritte) Kontakt (`n_contacts_mean=0`, `grasped_mean=0`). Der Blick auf den vollständigen Verlauf (`grasp_stability.csv`, alle 20 Checkpoints je Lauf) zeigt aber ein differenzierteres Bild:

| Lauf | Verlauf über 2 Mio. Schritte |
|---|---|
| `sparse/A2C` | Bei **900k–1.2M Schritten** tatsächlich `n_contacts_mean=1.0, grasped_mean=1.0, mean_reward=0.0` an 3 von 20 Checkpoints – die Policy **hat** zwischenzeitlich gelernt, Kontakt zu halten. Danach (ab 1.3M) durchgängig wieder `n_contacts=0`. |
| `sparse/DDPG` | Durchgängig `n_contacts=0` über alle 20 Checkpoints, keine Verbesserung erkennbar. |
| `sparse/TD3` | Durchgängig `n_contacts=0`, `mean_reward=-1.0` exakt und unverändert über alle 20 Checkpoints. |
| `phase1_2_3/DDPG` | `sim_unstable_mean=1.0` an **jedem einzelnen** der 20 Checkpoints, bereits ab dem ersten (100k). Die Policy kollabiert also sehr früh in einen dauerhaft instabilen Zustand und bleibt dort für die restlichen ~1.9 Mio. Schritte. |
| `phase1_2_3/TD3` | `mean_reward=-0.914766` **exakt identisch** an allen 20 Checkpoints – die Policy verändert sich über 2 Mio. Schritte praktisch nicht mehr (eingefroren in einem Fixpunkt, kein messbarer Lernfortschritt). |

**Einordnung:** Das ist der wichtigste Befund der gesamten bisherigen Ablationsreihe. "Einfach länger trainieren" ist **keine verlässliche Lösung**: A2C zeigt, dass die Aufgabe grundsätzlich lösbar ist (Kontakt wurde zwischenzeitlich erreicht), aber die gefundene Lösung ist instabil und geht durch weiteres Training wieder verloren (katastrophisches Vergessen – erwartbar bei On-Policy-Algorithmen ohne Mechanismus, gute Verhaltensweisen "festzuhalten"). DDPG und TD3 zeigen zwei unterschiedliche Arten von Stagnation: dauerhafter Politik-Kollaps direkt zu Beginn (DDPG/phase1_2_3) bzw. ein eingefrorenes, nicht mehr lernendes Optimum (TD3/phase1_2_3, byte-identischer Reward über 2 Mio. Schritte).

### Aktualisierte Empfehlung für Iteration 4

Die Trainingsskala ist damit als Haupt-Hebel widerlegt; die nächsten Schritte sollten woanders ansetzen:

1. **Checkpoint-Auswahl statt "letzter Checkpoint":** Da `sparse/A2C` zwischenzeitlich (900k–1.2M) erfolgreich war, sollte künftig das **beste** Zwischen-Checkpoint ausgewertet/gesichert werden, nicht nur der Endstand nach `total_timesteps` – aktuell speichert `pipeline.py` zwar periodische Checkpoints, wählt aber keinen "besten" aus. Kleine, gezielte Ergänzung an `GraspMetricsEvalCallback`.
2. **On-Policy-Instabilität gezielt angehen:** Für A2C insbesondere ein Mechanismus gegen katastrophisches Vergessen prüfen (kleinere Lernrate nach erstem Erfolg, oder Wechsel zu PPO mit seinem Trust-Region-Clipping, das genau dieses Problem abmildern soll).
3. **DDPG/TD3-Politik-Kollaps vorrangig vor weiterer Skalierung untersuchen:** Beide zeigen bereits bei 100k–200k Schritten dieselbe Pathologie wie bei 2 Mio. – mehr Training behebt sie nicht. Sinnvoller wäre, DDPG/TD3-spezifische Hyperparameter (Explorationsrauschen, `learning_starts`, Replay-Buffer-Größe) zu untersuchen, bevor weitere große Läufe mit diesen Algorithmen gefahren werden.
4. `reward_retention`/`reward_dynamic` weiterhin auf `retained_ratio > 0` gaten (siehe Kernbefund 2 aus Iteration 1) – bleibt unverändert relevant, sobald Kontakt zuverlässiger auftritt.

## Iteration 4: Best-Checkpoint-Tracking (Empfehlung 1 aus Iteration 3)

Umsetzung: `GraspMetricsEvalCallback` (`pipeline.py`) trackt jetzt bei jeder periodischen Evaluation den bislang besten Checkpoint und speichert ihn separat (`checkpoints/best_model.zip` + `checkpoints/best_checkpoint.json`). Score-Formel: `mean_reward - 2.0 * sim_unstable_mean` – eine Instabilitäts-Episode senkt den Score immer stärker ab, als der Reward-Bereich (±1) je ausgleichen könnte, damit ein durch Truncation "zufällig gut aussehender" Reward (siehe `sparse/A2C` in Iteration 3) nicht fälschlich als bester Checkpoint gilt. `summary.csv`/`grasp_stability.csv` enthalten jetzt zusätzlich `best_step` und alle `best_*`-Metriken neben den bisherigen Werten des letzten Checkpoints. Auch bei per Watchdog abgebrochenen Läufen wird der beste bis dahin gefundene Checkpoint aus der Datei nachträglich übernommen.

Erneuter Lauf über **alle 4 Profile**, 5 Algorithmen, 1 Seed, **400.000 Schritte** (2× Iteration-2-Skala, gewählt um im Zeitbudget eines Tages zu bleiben – Iteration 2 bei 200k brauchte real 3h50min, hochgerechnet ~7,7h bei 400k). Ergebnisse in `results/ablation/`; Iteration-2-Daten (200k) liegen weiterhin in `results/ablation_200k_iter2/`, Iteration-3-Daten (2M, reduzierter Umfang) in `results/ablation_2m_iter3/`.

```bash
python run_reward_ablation.py --total-timesteps 400000 --eval-freq 20000
```

### Laufstatus

10 von 20 Läufen (50 %) per Watchdog beendet – erneut eine hohe Quote, in derselben Größenordnung wie bei den 2-Mio.-Läufen aus Iteration 3, obwohl die Skala hier nur ein Fünftel davon beträgt. Timeouts verteilen sich über alle 4 Profile (auch `sparse`: PPO, SAC), nicht nur über die Shaped-Varianten – die Instabilitätsrate scheint also eher eine Eigenschaft von Algorithmus/Seed-Kombination auf diesem Environment zu sein als ein Nebeneffekt des Reward-Shapings.

### Kernbefund: Best-Checkpoint-Auswahl bestätigt – kein Lauf zeigt echten Erfolg

Über **alle 20 Läufe** (abgeschlossene wie abgebrochene) ist `best_n_contacts_mean` und `best_grasped_mean` exakt `0.0`. Das schließt die in Iteration 3 offene Frage: Der damalige `sparse/A2C`-Befund (`n_contacts=1.0` bei 900k–1.2M Schritten) war mit `sim_unstable_mean=1.0` verknüpft – also vermutlich ein Instabilitäts-Artefakt und kein echter gelernter Griff. Beispiel `sparse/A2C` in dieser Iteration: finaler Checkpoint zeigt `mean_reward=0.0` (sieht besser aus als −1), aber `sim_unstable_mean=1.0`; der von der Score-Funktion gewählte beste Checkpoint (Schritt 20.000, der allererste) hat `mean_reward=-1.0` bei `sim_unstable_mean=0.0` – schlechter aussehender, aber tatsächlich stabilerer Zustand. Die Score-Funktion funktioniert also wie beabsichtigt und liefert hier ein klareres, durch Instabilität unverfälschtes Bild.

**Damit ist jetzt mit einiger Sicherheit auszuschließen, dass ein bereits vorhandener, nur falsch ausgewerteter Erfolg in den bisherigen Daten "versteckt" war** – über vier Iterationen (200k, 2M, erneut 400k mit korrigierter Checkpoint-Auswahl) zeigt kein einziger der >40 Trainingsläufe einen instabilitätsbereinigten, echten Kontakt-/Grasp-Erfolg.

### Aktualisierte Empfehlung für Iteration 5

Die "vielleicht haben wir nur falsch ausgewertet"-Hypothese ist damit widerlegt; das Problem liegt an der Aufgabe/dem Trainingsaufbau selbst, nicht an der Metrik-Auswertung. Sinnvolle nächste Hebel:

1. **Curriculum/Warmstart statt reinem Zufall:** Episode testweise mit bereits leicht geschlossenen Fingern oder in Griffnähe startend, um der Policy einen realistischen Ausgangspunkt zu geben, statt Grasp-Verhalten aus komplett zufälliger Exploration entdecken zu müssen.
2. **Algorithmus-Hyperparameter statt Trainingsdauer:** Da weder 200k, 400k noch 2 Mio. Schritte einen Unterschied gemacht haben, liegt der Hebel vermutlich nicht mehr in "mehr Schritte", sondern in Explorationsrauschen/Netzwerkgröße/Lernraten der einzelnen Algorithmen.
3. **Instabilitätsrate (~50 % über alle Iterationen hinweg) als eigenständiges Problem behandeln**, unabhängig vom Reward-Shaping – z. B. mit mehreren Seeds prüfen, ob das reproduzierbar bei denselben Algorithmus/Profil-Kombinationen auftritt oder tatsächlich zufällig verteilt ist.

## Iteration 5: Warmstart-Curriculum (Empfehlung 1 aus Iteration 4)

**Umsetzung:** Neuer opt-in Konstruktor-Kwarg `warm_start_joint_targets: dict[str, float] | None` auf `BaseMJEnv` (und durchgereicht durch `GraspEnv`/`ShapedGraspEnv`), Default `None` (reproduziert das alte Verhalten bit-identisch). Bei jedem `reset()` werden die angegebenen Joints auf einen festen `qpos`-Zielwert gesetzt; da Position-Aktuatoren gegen ihren `ctrl`-Wert servoen, muss zusätzlich der `ctrl`-Wert des jeweils treibenden Aktuators auf denselben Zielwert gesetzt werden – sonst würde der allererste `mj_step` den Joint per Regelung zurück Richtung 0 ziehen und den Warmstart sofort wieder aufheben (siehe `_apply_warm_start` in `base_mjenv.py`).

**Kalibrierung der Zielpose:** Reine Fingerflexion (wie ursprünglich naheliegend angenommen) hat sich als wirkungslos bis kontraproduktiv erwiesen – ein Zero-Action-Rollout-Sweep (700 Schritte, Seeds 0–2) zeigte `grasped`-Anteil 19 % bei 0.3 rad Flexion und 0 % ab 0.7 rad (die Finger krümmen sich von der Box weg statt zu ihr hin). Ursache: Bei Reset ist die Hand nicht in unmittelbarer Kontaktnähe zur Box, sondern nur zufällig/lose in der Nähe – Fingerflexion allein bringt keinen Kontakt zustande. Ein zusätzlicher, empirisch ermittelter Positions-Offset der Hand-Basis (`ee_X=-0.04`, `ee_Y=+0.06`, über die linke Box) kombiniert mit 0.3 rad Fingerflexion brachte den entscheidenden Unterschied: `grasped`-Anteil 99.8 % über die volle Episode (vs. 27 % bei Standardpose, Seeds 0–2). Nebenbefund bei der Kalibrierung: Die `ee_X`/`ee_Y`-Slide-Joints sind **nicht weltachsen-ausgerichtet** – `ee_X` bewegt die Hand entlang Welt-`+y`, `ee_Y` entlang Welt-`+x`, `ee_Z` entlang Welt-`-z` (verifiziert per direkter `xpos`-Messung vor/nach Warmstart-Anwendung).

Drei neue Reward-Profile in `reward_profiles.py` (`sparse_warmstart`, `phase1_warmstart`, `phase1_2_3_warmstart`), die exakt den bisherigen sparse/phase1/phase1_2_3-Profilen entsprechen, aber mit dieser kalibrierten Startpose. Kein bestehendes Profil wurde verändert – Vergleichbarkeit zu Iteration 1–4 bleibt erhalten.

**Zusätzlicher Fix im selben Commit:** `_compute_terminal_stability_reward` (`shaped_grasp_env.py`) gatet jetzt alle drei Phase-3-Komponenten (Retention/Energie/Dynamic-Stability) auf `retained_ratio > 0` – die in Iteration 1 dokumentierte Lücke (Empfehlung 4), die bislang unverändert offen war. Ohne dieses Gate hätte `phase1_2_3_warmstart` sonst denselben kontaktunabhängigen Sockelbonus wie in Iteration 1 mit sich geführt.

**Lauf:** Alle 3 Warmstart-Profile × 5 Algorithmen, 1 Seed, 400.000 Schritte (gleiche Skala wie Iteration 4, für direkte Vergleichbarkeit). Ergebnisse in `results/ablation/`; Iteration-4-Daten (400k, alle 4 Original-Profile) gesichert unter `results/ablation_400k_iter4/`.

```bash
python run_reward_ablation.py --profiles sparse_warmstart phase1_warmstart phase1_2_3_warmstart --algorithms PPO SAC TD3 A2C DDPG --total-timesteps 400000 --eval-freq 20000
```

Gestartet am 2026-07-08 im Hintergrund (Log: `ablation_v5_warmstart.log`); Ergebnisse und Auswertung folgen in einer Aktualisierung dieses Abschnitts, sobald der Lauf abgeschlossen ist.
