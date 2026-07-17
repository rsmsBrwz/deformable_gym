# Iteration 6: PPO/SAC/A2C-Hyperparameter-Sweep

**Ausgangslage:** Fünf Ablationsiterationen (200k–2 Mio. Schritte, sparse/Phase-1/1+2/1+2+3, mit und ohne Warmstart-Curriculum, s. [REWARD_SHAPING_ABLATION.md](REWARD_SHAPING_ABLATION.md)) haben nie eine Policy hervorgebracht, die echtes, durch Training erworbenes Greifen zeigt. DDPG/TD3 zeigen dabei einen eigenständigen, bereits identifizierten Fehlermodus (Policy-Kollaps: eingefrorene, vollständig gesättigte Policy ab dem ersten Checkpoint). PPO/SAC/A2C kollabieren nicht, aber sie konvergieren auch nicht auf Kontakt – sie bleiben über alle bisherigen Iterationen hinweg flach bei `n_contacts≈0`. Alle bisherigen Läufe verwendeten durchgängig SB3-Standard-Hyperparameter (`pipeline.py`: `algo_class("MlpPolicy", train_env, verbose=0, seed=seed)`, keine `learning_rate`/`ent_coef`/`policy_kwargs`-Anpassung).

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

Gestartet am 2026-07-11 im Hintergrund (Logs: `ablation_v6b_sac_multiseed.log`, `ablation_v6b_sac_long.log`). Ergebnisse folgen in einer Aktualisierung dieses Abschnitts.
