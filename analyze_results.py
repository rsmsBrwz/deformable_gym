"""Aggregate and visualize pipeline.py results for a presentation.

Scans a results directory (as produced by pipeline.py) across all
env/algorithm/seed runs and produces:
  - summary_table.csv / summary_table.md: one row per env x algorithm,
    mean (+/- std over seeds) of the final reward/success/grasp-stability
    measures.
  - plots/*.png: bar-chart comparisons across algorithms, and training-curve
    plots of reward/success/grasp measures over training timesteps.
  - report.html: a single page combining the table and all plots, ready to
    open in a browser (or print to PDF) for a presentation.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir ./results --output-dir ./results/report
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# (column in summary.csv / grasp_stability.csv, human-readable label)
SUMMARY_METRICS = [
    ("mean_reward", "Reward (Test)"),
    ("success_rate", "Erfolgsrate"),
    ("mean_length", "Episodenlaenge"),
    ("n_contacts_mean", "Kontaktanzahl"),
    ("total_normal_force_mean", "Kontaktkraft (Summe)"),
    ("grasped_mean", "Binary Grasp State"),
    ("retained_ratio_mean", "Object Retained Ratio"),
    ("dynamic_max_displacement_mean", "Dynamic Stability: max. Verschiebung"),
    ("dynamic_final_speed_mean", "Dynamic Stability: Restgeschwindigkeit"),
    ("dynamic_settle_step_mean", "Beruhigungsdauer (Schritte)"),
    ("energy_kinetic_after_mean", "Energie (kinetisch, nach Pause)"),
    ("energy_retained_ratio_mean", "Energie-Verhaeltnis (nachher/vorher)"),
    ("action_saturation_mean", "Aktionssaettigung"),
    ("sim_unstable_mean", "Anteil instabiler Episoden"),
]

CURVE_METRICS = [
    ("mean_reward", "Reward (Test)"),
    ("success_rate", "Erfolgsrate"),
    ("n_contacts_mean", "Kontaktanzahl"),
    ("retained_ratio_mean", "Object Retained Ratio"),
    ("energy_retained_ratio_mean", "Energie-Verhaeltnis (nachher/vorher)"),
    ("action_saturation_mean", "Aktionssaettigung"),
    ("sim_unstable_mean", "Anteil instabiler Episoden"),
]


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add metrics computed from already-logged raw values, so they are
    available immediately for older result sets too (no pipeline.py rerun
    needed), unlike metrics that come from a new info-dict key."""
    energy_cols = {
        "energy_potential_before_mean",
        "energy_kinetic_before_mean",
        "energy_potential_after_mean",
        "energy_kinetic_after_mean",
    }
    if energy_cols.issubset(df.columns):
        before = df["energy_potential_before_mean"] + df["energy_kinetic_before_mean"]
        after = df["energy_potential_after_mean"] + df["energy_kinetic_after_mean"]
        df["energy_retained_ratio_mean"] = after / before.replace(0, np.nan)
    return df


def find_runs(results_dir: str) -> list[tuple[str, str, str, str]]:
    """Find every run directory produced by pipeline.py.

    Returns a list of (env, algorithm, seed, run_dir) tuples, identified by
    the presence of a grasp_stability.csv file.
    """
    pattern = os.path.join(results_dir, "*", "*", "seed*", "grasp_stability.csv")
    runs = []
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        seed_dir = os.path.basename(run_dir)
        algo_dir = os.path.dirname(run_dir)
        algorithm = os.path.basename(algo_dir)
        env_dir = os.path.dirname(algo_dir)
        env = os.path.basename(env_dir)
        seed = seed_dir.replace("seed", "")
        runs.append((env, algorithm, seed, run_dir))
    return runs


def load_summary(results_dir: str, runs: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Load the final-performance row for every run.

    Prefers results_dir/summary.csv (written once a full pipeline.py run
    finishes); for runs missing from it (e.g. an interrupted run), falls
    back to the last row of that run's own grasp_stability.csv.
    """
    summary_path = os.path.join(results_dir, "summary.csv")
    df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()

    # A row only counts as "complete" if it actually has a mean_reward (a run
    # that was killed by pipeline.py's watchdog before finishing is present
    # in summary.csv as a timed_out placeholder with no metrics at all --
    # those still need the grasp_stability.csv fallback below).
    complete_keys = set()
    if not df.empty and {"env", "algorithm", "seed"}.issubset(df.columns):
        has_metrics = df["mean_reward"].notna() if "mean_reward" in df.columns else False
        complete = df[has_metrics] if isinstance(has_metrics, pd.Series) else df.iloc[0:0]
        complete_keys = set(
            zip(complete["env"], complete["algorithm"], complete["seed"].astype(str))
        )
        df = df[df.apply(lambda r: (r["env"], r["algorithm"], str(r["seed"])) in complete_keys, axis=1)]

    fallback_rows = []
    for env, algorithm, seed, run_dir in runs:
        if (env, algorithm, seed) in complete_keys:
            continue
        gs_path = os.path.join(run_dir, "grasp_stability.csv")
        if not os.path.exists(gs_path):
            continue
        gs = pd.read_csv(gs_path)
        if gs.empty:
            continue
        row = gs.iloc[-1].to_dict()
        row.update(
            {"env": env, "algorithm": algorithm, "seed": int(seed), "timed_out": True}
        )
        fallback_rows.append(row)

    if fallback_rows:
        df = pd.concat([df, pd.DataFrame(fallback_rows)], ignore_index=True)
    if "timed_out" not in df.columns:
        df["timed_out"] = False
    df["timed_out"] = df["timed_out"].fillna(False)
    return add_derived_metrics(df)


def load_curves(runs: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Load grasp_stability.csv for every run into one long DataFrame,
    tagged with env/algorithm/seed, for plotting training curves."""
    frames = []
    for env, algorithm, seed, run_dir in runs:
        df = pd.read_csv(os.path.join(run_dir, "grasp_stability.csv"))
        df["env"] = env
        df["algorithm"] = algorithm
        df["seed"] = int(seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _ in SUMMARY_METRICS if m in summary_df.columns]
    if "num_timesteps" in summary_df.columns:
        metrics = ["num_timesteps"] + metrics
    grouped = summary_df.groupby(["env", "algorithm"])
    mean = grouped[metrics].mean()
    std = grouped[metrics].std()
    n_seeds = grouped.size().rename("n_seeds")
    multi_seed = n_seeds.max() > 1

    labels = dict(SUMMARY_METRICS)
    labels["num_timesteps"] = "Trainingsschritte erreicht"

    table = pd.DataFrame(index=mean.index)
    for metric in metrics:
        label = labels[metric]
        if multi_seed:
            table[label] = [
                f"{mu:.3g} +/- {sd:.3g}" if pd.notna(mu) else "n/a"
                for mu, sd in zip(mean[metric], std[metric])
            ]
        else:
            table[label] = [
                f"{mu:.3g}" if pd.notna(mu) else "n/a" for mu in mean[metric]
            ]
    table["n_seeds"] = n_seeds
    if "timed_out" in summary_df.columns:
        any_timed_out = summary_df.groupby(["env", "algorithm"])["timed_out"].any()
        table["Status"] = [
            "abgebrochen (Zeitlimit)" if t else "vollstaendig"
            for t in any_timed_out
        ]
    return table.reset_index()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def plot_bar_comparisons(summary_df: pd.DataFrame, plots_dir: str) -> list[str]:
    paths = []
    for metric, label in SUMMARY_METRICS:
        if metric not in summary_df.columns or summary_df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=summary_df, x="algorithm", y=metric, hue="env", errorbar="sd", ax=ax
        )
        ax.set_title(label)
        ax.set_xlabel("Algorithmus")
        ax.set_ylabel(label)
        fig.tight_layout()
        path = os.path.join(plots_dir, f"bar_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_training_curves(curves_df: pd.DataFrame, plots_dir: str) -> list[str]:
    paths = []
    for env in sorted(curves_df["env"].unique()):
        env_df = curves_df[curves_df["env"] == env]
        for metric, label in CURVE_METRICS:
            if metric not in env_df.columns or env_df[metric].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=env_df,
                x="num_timesteps",
                y=metric,
                hue="algorithm",
                marker="o",
                errorbar="sd",
                ax=ax,
            )
            ax.set_title(f"{label} ueber Trainingszeit ({env})")
            ax.set_xlabel("Trainingsschritte")
            ax.set_ylabel(label)
            fig.tight_layout()
            safe_env = env.replace("/", "_")
            path = os.path.join(plots_dir, f"curve_{safe_env}_{metric}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def build_html_report(
    output_dir: str,
    table: pd.DataFrame,
    bar_paths: list[str],
    curve_paths: list[str],
    results_dir: str,
) -> str:
    def rel(p: str) -> str:
        return os.path.relpath(p, output_dir)

    header = "".join(f"<th>{c}</th>" for c in table.columns)
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in table.itertuples(index=False)
    )
    bar_imgs = "\n".join(
        f'<figure><img src="{rel(p)}"><figcaption>{os.path.basename(p)}</figcaption></figure>'
        for p in bar_paths
    )
    curve_imgs = "\n".join(
        f'<figure><img src="{rel(p)}"><figcaption>{os.path.basename(p)}</figcaption></figure>'
        for p in curve_paths
    )

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Grasp-Training Ergebnisse</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; background: #fff; }}
  h1, h2 {{ color: #1a2b4c; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; white-space: nowrap; }}
  th {{ background: #1a2b4c; color: white; }}
  tr:nth-child(even) {{ background: #f4f6fa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; margin-bottom: 2em; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 0.75em; color: #888; text-align: center; }}
  img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 0.85em; }}
  .table-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
  <h1>Grasp-Training Ergebnisse</h1>
  <p class="meta">Erzeugt am {datetime.now().strftime("%Y-%m-%d %H:%M")} aus {os.path.abspath(results_dir)}</p>

  <h2>Zusammenfassung (ueber Seeds gemittelt)</h2>
  <div class="table-wrap">
    <table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
  </div>

  <h2>Vergleich der Algorithmen</h2>
  <div class="grid">{bar_imgs}</div>

  <h2>Trainingsverlauf</h2>
  <div class="grid">{curve_imgs}</div>
</body>
</html>
"""
    path = os.path.join(output_dir, "report.html")
    with open(path, "w") as f:
        f.write(html)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument(
        "--output-dir", default=None, help="Default: <results-dir>/report"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "report")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    runs = find_runs(args.results_dir)
    if not runs:
        print(
            f"Keine pipeline.py-Ergebnisse unter {args.results_dir} gefunden "
            "(erwartet wird */*/seed*/grasp_stability.csv)."
        )
        return

    summary_df = load_summary(args.results_dir, runs)
    curves_df = load_curves(runs)

    table = make_summary_table(summary_df)
    table.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    with open(os.path.join(output_dir, "summary_table.md"), "w") as f:
        f.write(dataframe_to_markdown(table))

    bar_paths = plot_bar_comparisons(summary_df, plots_dir)
    curve_paths = plot_training_curves(curves_df, plots_dir)

    report_path = build_html_report(output_dir, table, bar_paths, curve_paths, args.results_dir)

    print(table.to_string(index=False))
    print(f"\nReport:  {report_path}")
    print(f"Tabelle: {os.path.join(output_dir, 'summary_table.csv')} / .md")
    print(f"Plots:   {plots_dir} ({len(bar_paths)} Vergleichs-, {len(curve_paths)} Verlaufsplots)")


if __name__ == "__main__":
    main()
