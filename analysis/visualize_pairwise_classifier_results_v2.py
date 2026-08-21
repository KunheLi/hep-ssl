import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


PAIR_ORDER = [
    "ggf_vs_ttbar",
    "ggf_vs_dihiggs",
    "ttbar_vs_dihiggs",
]

PAIR_LABELS = {
    "ggf_vs_ttbar": r"ggF vs $t\bar{t}$",
    "ggf_vs_dihiggs": "ggF vs di-Higgs",
    "ttbar_vs_dihiggs": r"$t\bar{t}$ vs di-Higgs",
}

CLASSIFIER_ORDER = [
    "logistic_regression",
    "knn",
    "gradient_boosting",
]

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "k-Nearest Neighbors",
    "gradient_boosting": "Gradient Boosting",
}

AUGMENTATION_ORDER = [
    "none",
    "rex",
    "res",
    "rec",
    "rxs",
    "rxc",
    "rsc",
    "exs",
    "exc",
    "esc",
    "xsc",
    "rexs",
    "rexc",
    "resc",
    "rxsc",
    "exsc",
    "rexsc",
]

# Each tuple records the actual execution sequence, not an unordered set.
AUGMENTATION_STEPS = {
    "none": (),
    "rex": ("Rotation", "Energy Noise", "XYZ Noise"),
    "res": ("Rotation", "Energy Noise", "Spatial Shift"),
    "rec": ("Rotation", "Energy Noise", "Spatial Crop"),
    "rxs": ("Rotation", "XYZ Noise", "Spatial Shift"),
    "rxc": ("Rotation", "XYZ Noise", "Spatial Crop"),
    "rsc": ("Rotation", "Spatial Shift", "Spatial Crop"),
    "exs": ("Energy Noise", "XYZ Noise", "Spatial Shift"),
    "exc": ("Energy Noise", "XYZ Noise", "Spatial Crop"),
    "esc": ("Energy Noise", "Spatial Shift", "Spatial Crop"),
    "xsc": ("XYZ Noise", "Spatial Shift", "Spatial Crop"),
    "rexs": (
        "Rotation",
        "Energy Noise",
        "XYZ Noise",
        "Spatial Shift",
    ),
    "rexc": (
        "Rotation",
        "Energy Noise",
        "XYZ Noise",
        "Spatial Crop",
    ),
    "resc": (
        "Rotation",
        "Energy Noise",
        "Spatial Shift",
        "Spatial Crop",
    ),
    "rxsc": (
        "Rotation",
        "XYZ Noise",
        "Spatial Shift",
        "Spatial Crop",
    ),
    "exsc": (
        "Energy Noise",
        "XYZ Noise",
        "Spatial Shift",
        "Spatial Crop",
    ),
    "rexsc": (
        "Rotation",
        "Energy Noise",
        "XYZ Noise",
        "Spatial Shift",
        "Spatial Crop",
    ),
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
}

CLASSIFIER_COLORS = {
    "logistic_regression": "#4C78A8",
    "knn": "#F58518",
    "gradient_boosting": "#54A24B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and visualize 51 pretrained-encoder evaluations and "
            "three matched random-encoder controls."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Directory containing one extracted subdirectory per run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which organized tables and figures are written.",
    )
    parser.add_argument(
        "--expected-runs",
        default=54,
        type=int,
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png",),
        help=(
            "Figure formats to write. The default is PNG only; use "
            "'--formats png pdf' when vector PDFs are also needed."
        ),
    )
    return parser.parse_args()


def augmentation_display_name(configuration):
    if configuration == "random":
        return "Random Untrained Encoder"
    steps = AUGMENTATION_STEPS[configuration]
    if not steps:
        return "No Augmentation"
    return " → ".join(steps)


def augmentation_file_name(configuration):
    if configuration == "random":
        return "random_untrained_encoder"
    steps = AUGMENTATION_STEPS[configuration]
    if not steps:
        return "no_augmentation"
    return "_then_".join(step.lower().replace(" ", "_") for step in steps)


def wrapped_augmentation_name(configuration):
    """Wrap only for layout; never abbreviate an augmentation name."""
    if configuration in {"none", "random"}:
        return augmentation_display_name(configuration)
    steps = AUGMENTATION_STEPS[configuration]
    if len(steps) <= 3:
        return " → ".join(steps)
    return " → ".join(steps[:3]) + " →\n" + " → ".join(steps[3:])


def canonical_pair(channel_a, channel_b):
    pair = f"{channel_a}_vs_{channel_b}"
    if pair not in PAIR_ORDER:
        raise ValueError(
            f"Unexpected process pair '{pair}'. Expected one of {PAIR_ORDER}."
        )
    return pair


def configuration_from_run_name(run_name, channel_a, channel_b):
    prefix = f"{channel_a}_{channel_b}_"
    if not run_name.startswith(prefix):
        raise ValueError(
            f"Run directory '{run_name}' does not begin with '{prefix}'."
        )
    return run_name[len(prefix):]


def collect_metrics(results_dir, expected_runs):
    summary_paths = sorted(results_dir.glob("*/summary.json"))
    if len(summary_paths) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} summary.json files under {results_dir}, "
            f"but found {len(summary_paths)}."
        )

    rows = []
    seen_runs = set()

    for summary_path in summary_paths:
        run_name = summary_path.parent.name
        if run_name in seen_runs:
            raise RuntimeError(f"Duplicate run directory: {run_name}")
        seen_runs.add(run_name)

        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)

        channel_a = str(summary["channel_a"])
        channel_b = str(summary["channel_b"])
        pair = canonical_pair(channel_a, channel_b)
        configuration = configuration_from_run_name(
            run_name,
            channel_a,
            channel_b,
        )
        encoder_mode = (
            "random"
            if configuration == "random"
            or summary.get("encoder_mode") == "random"
            else "pretrained"
        )

        if encoder_mode == "random":
            configuration = "random"
        elif configuration not in AUGMENTATION_ORDER:
            raise ValueError(
                f"Unknown augmentation code '{configuration}' in {run_name}."
            )

        classifier_metrics = summary.get("classifiers", {})
        missing = set(CLASSIFIER_ORDER) - set(classifier_metrics)
        if missing:
            raise KeyError(
                f"{summary_path} is missing classifiers: {sorted(missing)}"
            )

        for classifier in CLASSIFIER_ORDER:
            metrics = classifier_metrics[classifier]
            row = {
                "source_experiment_id": run_name,
                "process_pair": pair,
                "process_pair_label": PAIR_LABELS[pair],
                "channel_a": channel_a,
                "channel_b": channel_b,
                "encoder_mode": encoder_mode,
                "source_augmentation_code": configuration,
                "augmentation_sequence": augmentation_display_name(
                    configuration
                ),
                "augmentation_file_name": augmentation_file_name(
                    configuration
                ),
                "classifier": classifier,
                "classifier_label": CLASSIFIER_LABELS[classifier],
            }
            for metric in METRIC_ORDER:
                value = metrics.get(metric, np.nan)
                row[metric] = float(value) if value is not None else np.nan
            rows.append(row)

    frame = pd.DataFrame(rows)
    expected_rows = expected_runs * len(CLASSIFIER_ORDER)
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} classifier rows, found {len(frame)}."
        )
    return frame


def add_random_baselines(frame):
    random_rows = frame[frame["encoder_mode"] == "random"].copy()
    expected = len(PAIR_ORDER) * len(CLASSIFIER_ORDER)
    if len(random_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} random-baseline rows, found "
            f"{len(random_rows)}."
        )

    columns = ["process_pair", "classifier"] + METRIC_ORDER
    baselines = random_rows[columns].rename(
        columns={metric: f"random_encoder_{metric}" for metric in METRIC_ORDER}
    )
    merged = frame.merge(
        baselines,
        on=["process_pair", "classifier"],
        how="left",
        validate="many_to_one",
    )
    for metric in METRIC_ORDER:
        merged[f"{metric}_gain_over_random_encoder"] = (
            merged[metric] - merged[f"random_encoder_{metric}"]
        )
    return merged


def add_sorting_and_ranks(frame):
    pair_order = {name: index for index, name in enumerate(PAIR_ORDER)}
    classifier_order = {
        name: index for index, name in enumerate(CLASSIFIER_ORDER)
    }
    augmentation_order = {
        **{name: index for index, name in enumerate(AUGMENTATION_ORDER)},
        "random": -1,
    }
    frame = frame.copy()
    frame["process_pair_order"] = frame["process_pair"].map(pair_order)
    frame["classifier_order"] = frame["classifier"].map(classifier_order)
    frame["augmentation_order"] = frame["source_augmentation_code"].map(
        augmentation_order
    )
    frame = frame.sort_values(
        ["process_pair_order", "classifier_order", "augmentation_order"]
    ).reset_index(drop=True)

    pretrained = frame["encoder_mode"] == "pretrained"
    for metric in ("accuracy", "roc_auc"):
        frame.loc[pretrained, f"{metric}_rank_within_pair_classifier"] = (
            frame.loc[pretrained]
            .groupby(["process_pair", "classifier"])[metric]
            .rank(method="min", ascending=False)
        )
    return frame


def save_figure(fig, directory, stem, formats):
    directory.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        fig.savefig(directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_limits(frame, pair, metric, delta):
    subset = frame[frame["process_pair"] == pair]
    if delta:
        column = f"{metric}_gain_over_random_encoder"
        values = subset[subset["encoder_mode"] == "pretrained"][column]
        limit = max(0.01, float(np.nanmax(np.abs(values))) * 1.18)
        return -limit, limit

    values = subset[metric]
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    lower = max(0.0, min(0.5, minimum - 0.04))
    upper = min(1.01, max(0.6, maximum + 0.04))
    return lower, upper


def plot_metric_for_pair(
    frame,
    pair,
    metric,
    directory,
    formats,
    delta=False,
):
    pretrained = frame[
        (frame["encoder_mode"] == "pretrained")
        & (frame["process_pair"] == pair)
    ]
    configurations = AUGMENTATION_ORDER
    y = np.arange(len(configurations))
    labels = [wrapped_augmentation_name(code) for code in configurations]
    column = f"{metric}_gain_over_random_encoder" if delta else metric
    xlabel = (
        f"{METRIC_LABELS[metric]} gain over matched random encoder"
        if delta
        else METRIC_LABELS[metric]
    )

    fig, axes = plt.subplots(
        1,
        len(CLASSIFIER_ORDER),
        figsize=(22, 13.5),
        sharey=True,
        constrained_layout=True,
    )
    xmin, xmax = metric_limits(frame, pair, metric, delta)

    for index, (ax, classifier) in enumerate(
        zip(axes, CLASSIFIER_ORDER)
    ):
        selected = pretrained[pretrained["classifier"] == classifier]
        values = (
            selected.set_index("source_augmentation_code")[column]
            .reindex(configurations)
            .to_numpy(dtype=float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Missing {column} values for {pair}, {classifier}."
            )

        if delta:
            colors = np.where(values >= 0.0, "#2A9D8F", "#E76F51")
            ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--")
            for ypos, value, color in zip(y, values, colors):
                ax.hlines(ypos, 0.0, value, color=color, linewidth=2)
                ax.scatter(value, ypos, color=color, s=48, zorder=3)
        else:
            baseline = float(
                frame[
                    (frame["process_pair"] == pair)
                    & (frame["classifier"] == classifier)
                    & (frame["encoder_mode"] == "random")
                ][metric].iloc[0]
            )
            ax.axvline(
                baseline,
                color="#777777",
                linewidth=1.4,
                linestyle="--",
                label=f"Random encoder = {baseline:.3f}",
            )
            ax.scatter(
                values,
                y,
                color=CLASSIFIER_COLORS[classifier],
                s=52,
                zorder=3,
            )
            ax.legend(loc="lower right", fontsize=9)

        for ypos, value in zip(y, values):
            label = f"{value:+.3f}" if delta else f"{value:.3f}"
            alignment = "left" if value >= 0 or not delta else "right"
            offset = (xmax - xmin) * 0.008
            xtext = value + offset if alignment == "left" else value - offset
            ax.text(
                xtext,
                ypos,
                label,
                va="center",
                ha=alignment,
                fontsize=8,
            )

        ax.set_title(CLASSIFIER_LABELS[classifier], fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_ylabel("Augmentation sequence used during pretraining")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.10)

    axes[0].invert_yaxis()
    comparison = (
        "Gain over the matched random untrained encoder"
        if delta
        else "Absolute downstream performance"
    )
    fig.suptitle(
        f"{PAIR_LABELS[pair]} — {METRIC_LABELS[metric]}\n{comparison}",
        fontsize=16,
        fontweight="bold",
    )
    suffix = (
        f"{metric}_gain_over_random_encoder"
        if delta
        else f"absolute_{metric}"
    )
    save_figure(
        fig,
        directory,
        f"{pair}__{suffix}_by_full_augmentation_sequence",
        formats,
    )


def best_pretrained_rows(frame, metric):
    pretrained = frame[frame["encoder_mode"] == "pretrained"]
    indices = pretrained.groupby(
        ["process_pair", "classifier"],
        sort=False,
    )[metric].idxmax()
    return pretrained.loc[indices].copy()


def load_predictions(results_dir, run_name, classifier):
    path = results_dir / run_name / classifier / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    with np.load(path, allow_pickle=True) as saved:
        labels = np.asarray(saved["test_labels"], dtype=np.int64)
        probabilities = np.asarray(saved["probabilities"])
    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        scores = probabilities[:, 1].astype(np.float64)
    elif probabilities.ndim == 1:
        scores = probabilities.astype(np.float64)
    else:
        raise ValueError(
            f"Unexpected probability shape {probabilities.shape} in {path}."
        )
    return labels, scores


def plot_selected_roc_curves(
    frame,
    results_dir,
    pair,
    directory,
    formats,
):
    best = best_pretrained_rows(frame, "roc_auc")
    fig, axes = plt.subplots(
        1,
        len(CLASSIFIER_ORDER),
        figsize=(18, 5.8),
        constrained_layout=True,
    )

    for ax, classifier in zip(axes, CLASSIFIER_ORDER):
        subset = frame[
            (frame["process_pair"] == pair)
            & (frame["classifier"] == classifier)
        ]
        random_row = subset[subset["encoder_mode"] == "random"].iloc[0]
        none_row = subset[
            (subset["encoder_mode"] == "pretrained")
            & (subset["source_augmentation_code"] == "none")
        ].iloc[0]
        best_row = best[
            (best["process_pair"] == pair)
            & (best["classifier"] == classifier)
        ].iloc[0]

        candidates = [
            ("Random Untrained Encoder", random_row, "#777777", "--"),
            ("No Augmentation", none_row, "#4C78A8", "-"),
            (
                best_row["augmentation_sequence"],
                best_row,
                "#E45756",
                "-",
            ),
        ]
        plotted_runs = set()
        for label, row, color, linestyle in candidates:
            run_name = row["source_experiment_id"]
            if run_name in plotted_runs:
                continue
            plotted_runs.add(run_name)
            labels, scores = load_predictions(
                results_dir,
                run_name,
                classifier,
            )
            false_positive_rate, true_positive_rate, _ = roc_curve(
                labels,
                scores,
            )
            curve_auc = auc(false_positive_rate, true_positive_rate)
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label}\nROC-AUC = {curve_auc:.3f}",
            )

        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        ax.set_title(CLASSIFIER_LABELS[classifier], fontweight="bold")
        ax.set_xlabel("False-Positive Rate")
        ax.set_ylabel("True-Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.20)
        ax.legend(fontsize=7.8, loc="lower right")

    fig.suptitle(
        f"{PAIR_LABELS[pair]} — ROC curves\n"
        "Random encoder, no augmentation, and best augmentation sequence",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(
        fig,
        directory,
        f"{pair}__selected_roc_curves_with_full_augmentation_names",
        formats,
    )


def plot_best_accuracy_summary(frame, directory, formats):
    best = best_pretrained_rows(frame, "accuracy").copy()
    best = best.sort_values(
        ["process_pair_order", "classifier_order"]
    ).reset_index(drop=True)
    best["accuracy_gain"] = best["accuracy_gain_over_random_encoder"]

    y = np.arange(len(best))
    colors = [CLASSIFIER_COLORS[name] for name in best["classifier"]]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.barh(y, best["accuracy_gain"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1.2)

    ylabels = [
        f"{PAIR_LABELS[row.process_pair]} | "
        f"{CLASSIFIER_LABELS[row.classifier]}"
        for row in best.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()

    span = max(0.01, float(best["accuracy_gain"].abs().max()))
    for ypos, row in enumerate(best.itertuples(index=False)):
        value = row.accuracy_gain
        alignment = "left" if value >= 0 else "right"
        xtext = value + 0.02 * span if value >= 0 else value - 0.02 * span
        ax.text(
            xtext,
            ypos,
            f"{value:+.3f}  |  {row.augmentation_sequence}",
            va="center",
            ha=alignment,
            fontsize=8.5,
        )

    ax.set_xlabel("Best accuracy gain over matched random untrained encoder")
    ax.set_title(
        "Best augmentation sequence for every process pair and classifier",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(
        fig,
        directory,
        "best_accuracy_gain_with_full_augmentation_sequences",
        formats,
    )
    return best


def write_tables(frame, tables_dir):
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / "all_classifier_metrics.csv", index=False)

    pretrained = frame[frame["encoder_mode"] == "pretrained"].copy()
    pretrained.to_csv(
        tables_dir / "pretrained_classifier_metrics.csv",
        index=False,
    )
    frame[frame["encoder_mode"] == "random"].to_csv(
        tables_dir / "random_untrained_encoder_baselines.csv",
        index=False,
    )

    for metric in ("accuracy", "roc_auc"):
        ranking = pretrained.sort_values(
            ["process_pair_order", "classifier_order", metric],
            ascending=[True, True, False],
        )
        ranking.to_csv(
            tables_dir / f"{metric}_ranking_by_process_pair_and_classifier.csv",
            index=False,
        )

    sequence_rows = []
    for configuration in AUGMENTATION_ORDER:
        sequence_rows.append(
            {
                "source_augmentation_code": configuration,
                "complete_augmentation_sequence": (
                    augmentation_display_name(configuration)
                ),
                "file_safe_augmentation_sequence": (
                    augmentation_file_name(configuration)
                ),
            }
        )
    pd.DataFrame(sequence_rows).to_csv(
        tables_dir / "complete_augmentation_sequences.csv",
        index=False,
    )


def write_readme(output_dir, formats):
    format_text = ", ".join(name.upper() for name in formats)
    text = f"""# Pairwise classifier visualizations

All figure labels spell out each augmentation in its actual execution order.
The short sweep codes are retained only in CSV provenance columns.

## Directories

- `00_tables/`: complete metrics, rankings, baselines, and name mapping.
- `01_accuracy/absolute/`: downstream accuracy for all 17 pretrained encoders.
- `01_accuracy/gain_over_random_encoder/`: accuracy minus the matched random-encoder baseline.
- `02_roc_auc/absolute/`: ROC-AUC for all 17 pretrained encoders.
- `02_roc_auc/gain_over_random_encoder/`: ROC-AUC minus the matched random-encoder baseline.
- `03_roc_curves/`: random, no-augmentation, and best-pretrained ROC curves.
- `04_summary/`: a compact comparison of the best augmentation sequence in each setting.

Generated figure format(s): {format_text}.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    frame = collect_metrics(results_dir, args.expected_runs)
    frame = add_random_baselines(frame)
    frame = add_sorting_and_ranks(frame)

    tables_dir = output_dir / "00_tables"
    accuracy_absolute_dir = output_dir / "01_accuracy" / "absolute"
    accuracy_gain_dir = (
        output_dir / "01_accuracy" / "gain_over_random_encoder"
    )
    roc_auc_absolute_dir = output_dir / "02_roc_auc" / "absolute"
    roc_auc_gain_dir = (
        output_dir / "02_roc_auc" / "gain_over_random_encoder"
    )
    roc_curves_dir = output_dir / "03_roc_curves"
    summary_dir = output_dir / "04_summary"

    write_tables(frame, tables_dir)

    for pair in PAIR_ORDER:
        plot_metric_for_pair(
            frame,
            pair,
            "accuracy",
            accuracy_absolute_dir,
            args.formats,
            delta=False,
        )
        plot_metric_for_pair(
            frame,
            pair,
            "accuracy",
            accuracy_gain_dir,
            args.formats,
            delta=True,
        )
        plot_metric_for_pair(
            frame,
            pair,
            "roc_auc",
            roc_auc_absolute_dir,
            args.formats,
            delta=False,
        )
        plot_metric_for_pair(
            frame,
            pair,
            "roc_auc",
            roc_auc_gain_dir,
            args.formats,
            delta=True,
        )
        plot_selected_roc_curves(
            frame,
            results_dir,
            pair,
            roc_curves_dir,
            args.formats,
        )

    best = plot_best_accuracy_summary(frame, summary_dir, args.formats)
    best.to_csv(
        tables_dir / "best_accuracy_result_for_each_pair_and_classifier.csv",
        index=False,
    )
    write_readme(output_dir, args.formats)

    print(f"Loaded {frame['source_experiment_id'].nunique()} runs.")
    print(f"Collected {len(frame)} classifier evaluations.")
    print("All figure labels use complete augmentation sequences.")
    print(f"Organized visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
