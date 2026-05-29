from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import string


SKILL_LEVEL_RANGES = {
    "Beginner": range(1, 9),
    "Intermediate": range(9, 17),
    "Professional": range(17, 25),
}


AGE_GROUP_BINS = [0, 30, 50, 200]
AGE_GROUP_LABELS = ["<30", "30-50", ">50"]


EXPERIENCE_GROUP_MAP = {
    "1 to 6 months": "< 1 year",
    "6 months to 1 year": "< 1 year",
    "1 to 3 years": "1-5 years",
    "3 to 5 years": "1-5 years",
    "More than 10 years": "> 10 years",
}
EXPERIENCE_GROUP_LABELS = ["< 1 year", "1-5 years", "> 10 years"]


HAND_DOMINANCE_LABELS = ["Right", "Left"]


GROUP_ORDERS = {
    "Skill Level": ["Beginner", "Intermediate", "Professional"],
    "Gender": ["Male", "Female"],
    "Age Group": AGE_GROUP_LABELS,
    "Golf Experience": EXPERIENCE_GROUP_LABELS,
    "Hand Dominance": HAND_DOMINANCE_LABELS,
}


ALPHA = 0.05
IQR_MULTIPLIER = 1.5
BOX_COLORS = ["#A7C957", "#F2E8CF", "#BC4749"]
BOX_CENTER_SPACING = 0.85
BOX_WIDTH = 0.5


TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 17
SKILL_LEVEL_TICK_LABEL_FONTSIZE = 13
N_LABEL_FONTSIZE = 15
SIGNIFICANCE_FONTSIZE = 20


@dataclass
class GroupTestResult:
    characteristic: str
    normality_per_group: dict[str, tuple[float, float]]
    all_normal: bool
    test_name: str
    test_statistic: float
    test_pvalue: float
    posthoc: pd.DataFrame | None
    descriptive: pd.DataFrame


def load_data(metadata_path: Path, annotation_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    annotation = pd.read_csv(annotation_path, encoding="utf-8-sig")

    metadata["pid"] = metadata["Participant Number"].str.replace("Sub", "").astype(int)
    annotation = annotation.rename(columns={"Participant Number": "pid"})

    annotation["Carry Distance (m)"] = pd.to_numeric(
        annotation["Carry Distance (m)"], errors="coerce"
    )
    annotation = annotation.dropna(subset=["Carry Distance (m)"])

    aggregated = _aggregate_per_participant(annotation)

    merged = aggregated.merge(
        metadata[
            ["pid", "Age", "Gender", "Hand Dominance", "Years of Golfing Experience"]
        ],
        on="pid",
        how="left",
    )

    merged["Skill Level"] = merged["pid"].apply(_assign_skill_level)
    merged["Age Group"] = pd.cut(
        merged["Age"], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS, right=False
    )
    merged["Golf Experience"] = merged["Years of Golfing Experience"].map(
        EXPERIENCE_GROUP_MAP
    )

    return merged


def _aggregate_per_participant(annotation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in annotation.groupby("pid"):
        values = group["Carry Distance (m)"].values
        n_before = len(values)
        cleaned = _remove_iqr_outliers(values)
        n_after = len(cleaned)
        rows.append(
            {
                "pid": pid,
                "Carry Distance (m)": np.mean(cleaned),
                "n_swings_total": n_before,
                "n_swings_used": n_after,
                "n_swings_removed": n_before - n_after,
            }
        )
    return pd.DataFrame(rows)


def _remove_iqr_outliers(values: np.ndarray) -> np.ndarray:
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
    return values[(values >= lower) & (values <= upper)]


def _assign_skill_level(pid: int) -> str:
    for level, id_range in SKILL_LEVEL_RANGES.items():
        if pid in id_range:
            return level
    raise ValueError(f"Participant ID {pid} does not belong to any skill group")


def describe_groups(
    df: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]
) -> pd.DataFrame:

    rows = []
    for group in group_order:
        values = df.loc[df[group_col] == group, value_col].dropna()
        if values.empty:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        rows.append(
            {
                "group": group,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median": values.median(),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
            }
        )
    return pd.DataFrame(rows)


def test_normality(
    df: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]
) -> dict[str, tuple[float, float]]:

    results = {}
    for group in group_order:
        values = df.loc[df[group_col] == group, value_col].dropna().values
        if len(values) < 3:
            results[group] = (np.nan, np.nan)
            continue
        w_stat, p_value = stats.shapiro(values)
        results[group] = (w_stat, p_value)
    return results


def compare_groups(
    df: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]
) -> GroupTestResult:

    normality = test_normality(df, group_col, value_col, group_order)

    all_normal = all(not np.isnan(p) and p > ALPHA for _, p in normality.values())

    group_values = [
        df.loc[df[group_col] == g, value_col].dropna().values
        for g in group_order
        if df[group_col].eq(g).any()
    ]
    present_groups = [g for g in group_order if df[group_col].eq(g).any()]

    n_groups = len(group_values)
    posthoc: pd.DataFrame | None = None

    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for {group_col}, got {n_groups}")

    if n_groups == 2:
        if all_normal:
            test_name = "Independent t-test (Welch)"
            stat, pval = stats.ttest_ind(*group_values, equal_var=False)
        else:
            test_name = "Mann-Whitney U test"
            stat, pval = stats.mannwhitneyu(*group_values, alternative="two-sided")
    else:
        if all_normal:
            test_name = "One-way ANOVA"
            stat, pval = stats.f_oneway(*group_values)
        else:
            test_name = "Kruskal-Wallis test"
            stat, pval = stats.kruskal(*group_values)

        if pval < ALPHA:
            posthoc = _pairwise_posthoc(
                df, group_col, value_col, present_groups, all_normal
            )

    descriptive = describe_groups(df, group_col, value_col, present_groups)

    return GroupTestResult(
        characteristic=group_col,
        normality_per_group=normality,
        all_normal=all_normal,
        test_name=test_name,
        test_statistic=float(stat),
        test_pvalue=float(pval),
        posthoc=posthoc,
        descriptive=descriptive,
    )


def _pairwise_posthoc(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: list[str],
    all_normal: bool,
) -> pd.DataFrame:

    n_comparisons = len(groups) * (len(groups) - 1) // 2
    p_matrix = pd.DataFrame(
        np.ones((len(groups), len(groups))),
        index=groups,
        columns=groups,
    )
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i >= j:
                continue
            v1 = df.loc[df[group_col] == g1, value_col].dropna().values
            v2 = df.loc[df[group_col] == g2, value_col].dropna().values
            if all_normal:
                _, p = stats.ttest_ind(v1, v2, equal_var=False)
            else:
                _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            p_corrected = min(p * n_comparisons, 1.0)
            p_matrix.loc[g1, g2] = p_corrected
            p_matrix.loc[g2, g1] = p_corrected
    return p_matrix


def plot_boxplots(
    df: pd.DataFrame, value_col: str, results: list[GroupTestResult], output_path: Path
) -> None:

    sns.set_style("whitegrid")

    for idx, result in enumerate(results):
        group_col = result.characteristic
        present_groups = result.descriptive["group"].tolist()
        plot_df = df[df[group_col].isin(present_groups)].copy()
        plot_df[group_col] = pd.Categorical(
            plot_df[group_col], categories=present_groups, ordered=True
        )
        x_positions = _box_positions(len(present_groups))

        fig_width = max(4.0, 1.0 * len(present_groups) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        grouped_values = [
            plot_df.loc[plot_df[group_col] == group, value_col].dropna().values
            for group in present_groups
        ]
        box = ax.boxplot(
            grouped_values,
            positions=x_positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], BOX_COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
        for element in ("whiskers", "caps", "medians"):
            for artist in box[element]:
                artist.set_color("black")

        mean_values = plot_df.groupby(group_col, observed=True)[value_col].mean()
        ax.scatter(
            x_positions,
            [mean_values.loc[group] for group in present_groups],
            marker="x",
            s=36,
            color="dimgray",
            alpha=0.65,
            linewidths=1.2,
            zorder=3,
        )

        ax.set_title(
            f"({string.ascii_lowercase[idx]}) {group_col}",
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel(value_col, fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        x_tick_fontsize = (
            SKILL_LEVEL_TICK_LABEL_FONTSIZE
            if group_col == "Skill Level"
            else TICK_LABEL_FONTSIZE
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(present_groups, fontsize=x_tick_fontsize)
        ax.set_xlim(x_positions[0] - 0.55, x_positions[-1] + 0.55)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        significant_pairs = _get_significant_pairs(result, present_groups)
        _set_boxplot_ylim(ax, plot_df[value_col], len(significant_pairs))
        _add_significance_annotations(ax, significant_pairs, x_positions)

        for x, group in zip(x_positions, present_groups):
            n = result.descriptive.loc[result.descriptive["group"] == group, "n"].iloc[
                0
            ]
            ax.annotate(
                f"n = {n}",
                xy=(x, ax.get_ylim()[0]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=N_LABEL_FONTSIZE,
                color="gray",
            )

        fig.tight_layout()
        figure_path = output_path.with_name(
            f"{output_path.stem}_{_slugify(group_col)}.png"
        )
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _get_significant_pairs(
    result: GroupTestResult, groups: list[str]
) -> list[tuple[int, int, float]]:

    if len(groups) == 2:
        if result.test_pvalue < ALPHA:
            return [(0, 1, result.test_pvalue)]
        return []

    if result.posthoc is None:
        return []

    pairs = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i >= j:
                continue
            p_value = result.posthoc.loc[g1, g2]
            if p_value < ALPHA:
                pairs.append((i, j, float(p_value)))
    return pairs


def _box_positions(n_groups: int) -> np.ndarray:
    return np.arange(n_groups, dtype=float) * BOX_CENTER_SPACING


def _set_boxplot_ylim(ax: plt.Axes, values: pd.Series, n_annotations: int) -> None:
    y_min = float(values.min())
    y_max = float(values.max())
    y_range = y_max - y_min
    if y_range == 0:
        y_range = max(abs(y_max) * 0.1, 1.0)

    lower_padding = 0.08 * y_range
    upper_padding = (0.16 + 0.10 * n_annotations) * y_range
    ax.set_ylim(y_min - lower_padding, y_max + upper_padding)


def _add_significance_annotations(
    ax: plt.Axes,
    pairs: list[tuple[int, int, float]],
    x_positions: np.ndarray,
) -> None:

    if not pairs:
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    bracket_height = 0.025 * y_range
    y_start = y_max - (0.13 + 0.10 * (len(pairs) - 1)) * y_range
    y_step = 0.10 * y_range

    for level, (x1, x2, p_value) in enumerate(pairs):
        x1_pos = x_positions[x1]
        x2_pos = x_positions[x2]
        y = y_start + level * y_step
        ax.plot(
            [x1_pos, x1_pos, x2_pos, x2_pos],
            [y, y + bracket_height, y + bracket_height, y],
            color="black",
            linewidth=1,
        )
        ax.text(
            (x1_pos + x2_pos) / 2,
            y + bracket_height,
            _p_to_stars(p_value),
            ha="center",
            va="bottom",
            color="black",
            fontsize=SIGNIFICANCE_FONTSIZE,
            fontweight="bold",
        )


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _slugify(label: str) -> str:
    return label.lower().replace(" ", "_")


def _format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def write_report(
    df: pd.DataFrame, results: list[GroupTestResult], output_path: Path
) -> None:

    lines = [
        "Carry Distance Analysis Report",
        "=" * 60,
        "",
        "Aggregation: per-participant mean after 1.5 x IQR outlier removal",
        f"Total participants analyzed: {len(df)}",
        f"Significance threshold (alpha): {ALPHA}",
        "Normality assessed by Shapiro-Wilk test.",
        "",
        "Per-participant swing counts (used / total):",
    ]
    for _, row in df.sort_values("pid").iterrows():
        lines.append(
            f"  Sub{int(row['pid']):02d}: "
            f"{int(row['n_swings_used'])}/{int(row['n_swings_total'])} "
            f"(removed {int(row['n_swings_removed'])}); "
            f"mean carry = {row['Carry Distance (m)']:.2f} m"
        )
    lines.append("")

    for result in results:
        lines.append(f"[{result.characteristic}]")
        lines.append("-" * 60)

        lines.append("Descriptive statistics:")
        for _, row in result.descriptive.iterrows():
            sd_str = f"{row['sd']:>6.2f}" if not np.isnan(row["sd"]) else "  N/A"
            lines.append(
                f"  {row['group']:<15} n={int(row['n']):<3}  "
                f"mean={row['mean']:>6.2f}  sd={sd_str}  "
                f"median={row['median']:>6.2f}  "
                f"IQR=[{row['q1']:.2f}, {row['q3']:.2f}]"
            )

        if len(result.descriptive) == 2:
            sizes = result.descriptive["n"].tolist()
            if min(sizes) <= 5 or max(sizes) / min(sizes) >= 3:
                lines.append(
                    f"  [Caveat] Group sizes are small or imbalanced "
                    f"({sizes[0]} vs {sizes[1]}); "
                    "interpret results with caution."
                )

        lines.append("Shapiro-Wilk normality:")
        for group, (w, p) in result.normality_per_group.items():
            if np.isnan(p):
                lines.append(f"  {group:<15} insufficient sample size")
                continue
            verdict = "normal" if p > ALPHA else "non-normal"
            lines.append(f"  {group:<15} W={w:.3f}, p={_format_p(p)} ({verdict})")
        lines.append(
            f"All groups normal: {result.all_normal} " f"-> using {result.test_name}"
        )

        lines.append(
            f"Test statistic = {result.test_statistic:.3f}, "
            f"p = {_format_p(result.test_pvalue)}"
        )

        if result.posthoc is not None:
            lines.append("Pairwise post-hoc (Bonferroni-corrected):")
            for i, g1 in enumerate(result.posthoc.index):
                for j, g2 in enumerate(result.posthoc.columns):
                    if i >= j:
                        continue
                    p = result.posthoc.loc[g1, g2]
                    sig = " *" if p < ALPHA else ""
                    lines.append(f"  {g1} vs {g2}: p = {_format_p(p)}{sig}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carry distance analysis by subject characteristics "
        "(per-participant aggregation)."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("Data") / "Documentation" / "Participant Metadata.csv",
        help="Path to Participant Metadata CSV.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("Data") / "Documentation" / "Annotation Data.csv",
        help="Path to Annotation Data CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save figure and report (default: results/).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.metadata, args.annotation)
    value_col = "Carry Distance (m)"

    characteristics = list(GROUP_ORDERS.keys())

    characteristics = [c for c in characteristics if df[c].nunique(dropna=True) >= 2]

    results = [
        compare_groups(df, c, value_col, GROUP_ORDERS[c]) for c in characteristics
    ]

    figure_path = args.output_dir / "carry_distance_boxplots.png"
    report_path = args.output_dir / "carry_distance_report.txt"

    plot_boxplots(df, value_col, results, figure_path)
    write_report(df, results, report_path)

    print("Figures saved: " f"{figure_path.with_name(figure_path.stem + '_*.png')}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
