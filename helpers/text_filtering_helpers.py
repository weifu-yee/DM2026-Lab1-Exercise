"""Term-document matrix filtering utilities for pattern mining."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def plot_filtering_distribution(
    axes,
    idx,
    category,
    metric_series,
    lower_threshold,
    upper_threshold,
    lower_pct,
    upper_pct,
    metric_name,
    color,
    words_to_keep,
    original_count,
):
    """Universal histogram plotting for any filtering type."""
    ax = axes[idx]

    x_max = np.percentile(metric_series, 99)
    bin_edges = np.linspace(0, x_max, 101)

    ax.hist(metric_series, bins=bin_edges, color=color, edgecolor="black", alpha=0.7)

    if lower_threshold is not None:
        ax.axvline(
            lower_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"{lower_pct}th %ile: {lower_threshold:.4f}",
        )
    if upper_threshold is not None:
        ax.axvline(
            upper_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"{upper_pct}th %ile: {upper_threshold:.4f}",
        )

    ax.set_xlabel(metric_name, fontsize=10)
    ax.set_ylabel("Number of Terms", fontsize=10)
    ax.set_title(
        f"{category}\n{original_count} → {len(words_to_keep)} terms",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)


def save_filtered_terms(
    output_dir,
    category,
    words_to_keep,
    words_removed_low,
    words_removed_high,
    metric_series,
    metric_name,
    lower_threshold,
    upper_threshold,
    filter_type,
):
    """Save kept and removed terms to text files."""
    category_safe = category.replace(".", "_")
    os.makedirs(f"{output_dir}/filtered_terms", exist_ok=True)

    with open(f"{output_dir}/filtered_terms/{category_safe}_kept_terms.txt", "w", encoding="utf-8") as f:
        f.write(f"{filter_type.upper()} Filtering - Terms Kept for {category}\n")
        f.write(f"Total: {len(words_to_keep)} terms\n")
        if upper_threshold:
            f.write(
                f"Threshold: {lower_threshold:.6f} <= {metric_name} <= {upper_threshold:.6f}\n"
            )
        else:
            f.write(f"Threshold: {metric_name} >= {lower_threshold:.6f}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Term':<40} {metric_name:>15}\n")
        f.write("-" * 80 + "\n")
        for term in sorted(words_to_keep):
            f.write(f"{term:<40} {metric_series[term]:15.6f}\n")

    if len(words_removed_low) > 0:
        with open(f"{output_dir}/filtered_terms/{category_safe}_removed_low.txt", "w", encoding="utf-8") as f:
            f.write(f"{filter_type.upper()} Filtering - Terms Removed (Low) for {category}\n")
            f.write(f"Total: {len(words_removed_low)} terms\n")
            f.write(f"Threshold: {metric_name} < {lower_threshold:.6f}\n")
            f.write("=" * 80 + "\n\n")
            for term in sorted(words_removed_low):
                f.write(f"{term}\n")

    if words_removed_high is not None and len(words_removed_high) > 0:
        with open(f"{output_dir}/filtered_terms/{category_safe}_removed_high.txt", "w", encoding="utf-8") as f:
            f.write(f"{filter_type.upper()} Filtering - Terms Removed (High) for {category}\n")
            f.write(f"Total: {len(words_removed_high)} terms\n")
            f.write(f"Threshold: {metric_name} > {upper_threshold:.6f}\n")
            f.write("=" * 80 + "\n\n")
            for term in sorted(words_removed_high):
                f.write(f"{term}\n")


def _save_filtering_summary(output_dir, summary_lines, filter_name):
    with open(f"{output_dir}/filtering_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{filter_name} Filtering Summary\n" + "=" * 70 + "\n\n")
        f.write(f"{'Category':<30} {'Original':>10} {'Filtered':>10} {'Removed':>10}\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")


def _apply_generic_filtering(
    filt_term_document_dfs,
    categories,
    output_dir,
    filter_type,
    metric_name,
    color,
    compute_metric_fn,
    compute_thresholds_fn,
    filter_words_fn,
    title_suffix,
):
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    filtered_dfs = {}
    summary_lines = []

    for idx, category in enumerate(categories):
        df_category = filt_term_document_dfs[category]
        metric_series = compute_metric_fn(df_category)
        lower_threshold, upper_threshold, lower_pct, upper_pct = compute_thresholds_fn(
            metric_series
        )
        words_to_keep, words_removed_low, words_removed_high = filter_words_fn(
            metric_series, lower_threshold, upper_threshold
        )

        filtered_dfs[category] = df_category[words_to_keep]

        plot_filtering_distribution(
            axes,
            idx,
            category,
            metric_series,
            lower_threshold,
            upper_threshold,
            lower_pct,
            upper_pct,
            metric_name,
            color,
            words_to_keep,
            len(df_category.columns),
        )

        save_filtered_terms(
            output_dir,
            category,
            words_to_keep,
            words_removed_low,
            words_removed_high,
            metric_series,
            metric_name,
            lower_threshold,
            upper_threshold,
            filter_type,
        )

        summary_lines.append(
            f"{category:<30} {len(df_category.columns):10d} "
            f"{len(words_to_keep):10d} "
            f"{len(df_category.columns) - len(words_to_keep):10d}"
        )

    plt.suptitle(
        f"{filter_type.replace('_', ' ').title()} Filtering: {title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/figures/{filter_type}_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    _save_filtering_summary(output_dir, summary_lines, filter_type.replace("_", " ").title())

    return filtered_dfs


def apply_variance_filtering(filt_term_document_dfs, categories, lower_pct=5, upper_pct=95):
    def compute_metric(df):
        return df.var(axis=0).sort_values(ascending=False)

    def compute_thresholds(metric_series):
        lower = np.percentile(metric_series, lower_pct)
        upper = np.percentile(metric_series, upper_pct)
        return lower, upper, lower_pct, upper_pct

    def filter_words(metric_series, lower, upper):
        kept = metric_series[(metric_series >= lower) & (metric_series <= upper)].index
        removed_low = metric_series[metric_series < lower].index
        removed_high = metric_series[metric_series > upper].index
        return kept, removed_low, removed_high

    return _apply_generic_filtering(
        filt_term_document_dfs,
        categories,
        "./output_files/variance",
        "variance",
        "Variance",
        "skyblue",
        compute_metric,
        compute_thresholds,
        filter_words,
        f"Middle {upper_pct - lower_pct}%",
    )


def apply_tfidf_filtering(filt_term_document_dfs, categories, threshold_pct=20):
    def compute_metric(df):
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(df.values)
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        return pd.Series(tfidf_scores, index=df.columns)

    def compute_thresholds(metric_series):
        threshold = np.percentile(metric_series, threshold_pct)
        return threshold, None, threshold_pct, None

    def filter_words(metric_series, lower, upper):
        kept = metric_series[metric_series >= lower].index
        removed_low = metric_series[metric_series < lower].index
        return kept, removed_low, None

    return _apply_generic_filtering(
        filt_term_document_dfs,
        categories,
        "./output_files/tfidf",
        "tfidf",
        "TF-IDF Score",
        "lightgreen",
        compute_metric,
        compute_thresholds,
        filter_words,
        f"Top {100 - threshold_pct}%",
    )


def apply_term_frequency_filtering(
    filt_term_document_dfs, categories, lower_pct=5, upper_pct=95
):
    def compute_metric(df):
        return df.sum(axis=0).sort_values(ascending=False)

    def compute_thresholds(metric_series):
        lower = np.percentile(metric_series, lower_pct)
        upper = np.percentile(metric_series, upper_pct)
        return lower, upper, lower_pct, upper_pct

    def filter_words(metric_series, lower, upper):
        kept = metric_series[(metric_series >= lower) & (metric_series <= upper)].index
        removed_low = metric_series[metric_series < lower].index
        removed_high = metric_series[metric_series > upper].index
        return kept, removed_low, removed_high

    return _apply_generic_filtering(
        filt_term_document_dfs,
        categories,
        "./output_files/term_freq",
        "term_freq",
        "Term Frequency",
        "lightcoral",
        compute_metric,
        compute_thresholds,
        filter_words,
        f"Middle {upper_pct - lower_pct}%",
    )
