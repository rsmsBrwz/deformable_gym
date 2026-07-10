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

Gestartet am 2026-07-10 im Hintergrund (Log: `ablation_v6_hparam_sweep.log`). Ergebnisse und Auswertung folgen in einer Aktualisierung dieses Dokuments, sobald der Lauf abgeschlossen ist.
