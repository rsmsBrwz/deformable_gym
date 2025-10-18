from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
import deformable_gym
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import sys
import stable_baselines3
from datetime import datetime

# Experimentkonfiguration
envs = ["SoftGym-ClothFold-v0", "SoftGym-RopeManipulation-v0", "SoftGym-LiquidPour-v0"]
env_configs = {
    "SoftGym-ClothFold-v0": {"friction": 0.5, "elasticity": 0.8},
    "SoftGym-RopeManipulation-v0": {"length": 2.0, "density": 0.3},
    "SoftGym-LiquidPour-v0": {"viscosity": 1.0, "pouring_speed": 0.2},
}
algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A3C": A2C, "DDPG": DDPG}
seeds = [0, 42, 100, 2023]  # Evtl. durch eine Rand-Funktion ersetzten und die Seeds wegspeicher
total_timesteps = 1_000_000

# Ergebnisordner
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

results = []

# Zusätzliche Metriken und Protokollierung

def get_grasp_stability(env):
    # Beispiel: Extrahiere Grasp Stability Measure aus der Umgebung, falls verfügbar
    # Platzhalter: Gibt None zurück
    if hasattr(env, 'get_grasp_stability'):
        return env.get_grasp_stability()
    return None

def get_success(env, obs):
    # Erfolgskriterium: Objekt wurde gehoben/greifen (z.B. Z-Position > Schwelle)
    if hasattr(env, 'get_object_position'):
        obj_pos = env.get_object_position()
        return obj_pos[2] > 0.05  # Schwelle anpassen
    return False

# Protokollierung der Versionen
run_metadata = {
    "python_version": sys.version,
    "stable_baselines3_version": stable_baselines3.__version__,
    "gymnasium_version": gym.__version__,
    "deformable_gym_version": getattr(deformable_gym, '__version__', 'unknown'),
    "run_time": datetime.now().isoformat(),
}

def save_plot(fig, metric, environment, algorithm, seed, results_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = os.path.join(results_dir, "plots", environment, algorithm)
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{metric}_seed{seed}_{timestamp}.png"
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    # Metadaten speichern
    meta = {
        "metric": metric,
        "environment": environment,
        "algorithm": algorithm,
        "seed": seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(plot_dir, f"{metric}_seed{seed}_{timestamp}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return filepath

# Test-Parameter für jede Umgebung
# Beispiel: Variiere Friktion, Elastizität, Objektform etc. im Test
# Diese Werte sollten sich von den Trainingsparametern unterscheiden und systematisch gewählt werden

test_env_configs = {
    "SoftGym-ClothFold-v0": {"friction": 0.8, "elasticity": 0.5},
    "SoftGym-RopeManipulation-v0": {"length": 2.5, "density": 0.5},
    "SoftGym-LiquidPour-v0": {"viscosity": 2.0, "pouring_speed": 0.1},
}

# Intervall für Policy-Evaluation während des Trainings
EVAL_INTERVAL = 1000  # Kann als Parameter übergeben werden
N_EVAL_EPISODES = 20  # Anzahl Testepisoden pro Checkpoint

for env_name in envs:
    for algo_name, algo_class in algorithms.items():
        for seed in seeds:
            train_param_set = env_configs.get(env_name, {})
            test_param_set = test_env_configs.get(env_name, {})
            print(f"Running {algo_name} on {env_name} with seed {seed} and params {train_param_set} (train) / {test_param_set} (test)")

            # Training-Environment initialisieren
            train_env = make_vec_env(env_name, seed=seed)
            # Falls Parameter gesetzt werden können, hier ergänzen
            # z.B. train_env.envs[0].set_params(**train_param_set)

            # Test-Environment initialisieren
            test_env = make_vec_env(env_name, seed=seed+999)
            # z.B. test_env.envs[0].set_params(**test_param_set)

            model = algo_class("MlpPolicy", train_env, verbose=0, seed=seed)

            # Für Visualisierung: Liste der Test-Performance über die Zeit
            eval_steps = []
            eval_rewards = []
            eval_success = []
            eval_grasp_stability = []
            eval_lengths = []

            # Trainingsschleife mit regelmäßiger Evaluation
            n_steps = 0
            while n_steps < total_timesteps:
                next_steps = min(EVAL_INTERVAL, total_timesteps - n_steps)
                model.learn(total_timesteps=next_steps, reset_num_timesteps=False)
                n_steps += next_steps

                # Policy speichern (optional)
                checkpoint_path = os.path.join(results_dir, "checkpoints", env_name, algo_name, f"seed{seed}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save(os.path.join(checkpoint_path, f"policy_step{n_steps}.zip"))

                # Policy auf Test-Environment evaluieren
                test_episode_rewards = []
                test_episode_success = []
                test_episode_lengths = []
                test_grasp_stabilities = []
                obs = test_env.reset()
                for ep in range(N_EVAL_EPISODES):
                    done = False
                    total_reward = 0
                    steps = 0
                    while not done:
                        action, _ = model.predict(obs)
                        obs, reward, done, info = test_env.step(action)
                        total_reward += reward
                        steps += 1
                    test_episode_rewards.append(total_reward)
                    test_episode_lengths.append(steps)
                    success = get_success(test_env, obs)
                    test_episode_success.append(success)
                    stability = get_grasp_stability(test_env)
                    test_grasp_stabilities.append(stability)
                    obs = test_env.reset()

                # Testmetriken speichern
                eval_steps.append(n_steps)
                eval_rewards.append(np.mean(test_episode_rewards))
                eval_success.append(np.mean(test_episode_success))
                eval_grasp_stability.append(np.nanmean(test_grasp_stabilities))
                eval_lengths.append(np.mean(test_episode_lengths))

            train_env.close()
            test_env.close()

            # Visualisierung der Test-Performance über die Trainingszeit
            plot_dir = os.path.join(results_dir, "plots", env_name, algo_name, f"seed{seed}")
            os.makedirs(plot_dir, exist_ok=True)
            def save_eval_plot(metric_values, metric_name, ylabel):
                fig = plt.figure(figsize=(10, 6))
                plt.plot(eval_steps, metric_values, marker='o')
                plt.title(f"{metric_name} über Trainingszeit: {algo_name} auf {env_name} (seed {seed})")
                plt.xlabel("Trainingsschritte")
                plt.ylabel(ylabel)
                plt.grid(True)
                filename = f"{metric_name}_eval_curve.png"
                filepath = os.path.join(plot_dir, filename)
                fig.savefig(filepath)
                plt.close(fig)
                # Metadaten speichern
                meta = {
                    "metric": metric_name,
                    "environment": env_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "train_params": train_param_set,
                    "test_params": test_param_set,
                    "eval_steps": eval_steps,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(os.path.join(plot_dir, f"{metric_name}_eval_curve_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                return filepath

            save_eval_plot(eval_rewards, "average_reward", "Durchschnittlicher Reward (Test)")
            save_eval_plot(eval_success, "success_rate", "Erfolgsrate (Test)")
            save_eval_plot(eval_grasp_stability, "grasp_stability_mean", "Grasp Stability (Test)")
            save_eval_plot(eval_lengths, "average_length", "Episodendauer (Test)")

# Ergebnisse als DataFrame speichern
results_df = pd.DataFrame(results)
results_file = os.path.join(results_dir, "experiment_results.csv")
results_df.to_csv(results_file, index=False)

# Metadaten speichern
with open(os.path.join(results_dir, "run_metadata.json"), "w") as f:
    json.dump(run_metadata, f, indent=2)

print(f"Results saved to {results_file}")


# Ergebnisse laden
results_df = pd.read_csv(results_file)

# Visualisierung
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="algorithm", y="average_reward", hue="environment")
plt.title("Performance verschiedener RL-Algorithmen")
plt.xlabel("Algorithmus")
plt.ylabel("Durchschnittlicher Reward")
plt.legend(title="Umgebung")
plt.show()

# Erweiterte Visualisierung
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="algorithm", y="success_rate", hue="environment")
plt.title("Erfolgsrate verschiedener RL-Algorithmen")
plt.xlabel("Algorithmus")
plt.ylabel("Erfolgsrate")
plt.legend(title="Umgebung")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="algorithm", y="grasp_stability_mean", hue="environment")
plt.title("Grasp Stability verschiedener RL-Algorithmen")
plt.xlabel("Algorithmus")
plt.ylabel("Grasp Stability (Mean)")
plt.legend(title="Umgebung")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="algorithm", y="average_length", hue="environment")
plt.title("Durchschnittliche Episodendauer")
plt.xlabel("Algorithmus")
plt.ylabel("Episodendauer (Schritte)")
plt.legend(title="Umgebung")
plt.show()

# Abschließende Gesamtplots (über alle Runs)
for metric in ["average_reward", "success_rate", "grasp_stability_mean", "average_length"]:
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="algorithm", y=metric, hue="environment")
    plt.title(f"{metric.replace('_', ' ').title()} verschiedener RL-Algorithmen")
    plt.xlabel("Algorithmus")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend(title="Umgebung")
    save_plot(fig, metric, "ALL", "ALL", "ALL", results_dir)
