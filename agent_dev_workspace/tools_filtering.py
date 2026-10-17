from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: Union[int, float] = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates population variance (ddof=0) for each feature in session.feature_matrix.
        Features with variance strictly greater than threshold (> threshold) are kept.

        Args:
            threshold: Minimum variance threshold. Features with variance > threshold are kept. Default is 0.0.

        Returns:
            Dict containing result_id, threshold, total_features, kept_features_count,
            removed_features_count, and all_zero_variance flag.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be initialized first using build_dtm_tool.")

        X = session.feature_matrix
        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError("session.feature_matrix has 0 documents; cannot compute variance.")

        # Compute population variance directly on sparse matrix without dense conversion
        # Var(X) = E[X^2] - (E[X])^2
        mean = np.array(X.mean(axis=0)).ravel()
        squared_mean = np.array(X.power(2).mean(axis=0)).ravel()
        variances = squared_mean - np.square(mean)
        # Ensure non-negative variances due to potential floating point precision
        variances = np.maximum(variances, 0.0)

        feature_names = np.array(session.feature_names)

        # Features with variance strictly > threshold are kept
        kept_mask = variances > threshold
        kept_features_count = int(np.sum(kept_mask))
        removed_features_count = len(feature_names) - kept_features_count

        # Construct full report DataFrame: feature as 1st column, variance as 2nd column, sorted descending
        report_df = pd.DataFrame({
            "feature": feature_names,
            "variance": variances,
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_features_count": kept_features_count,
            "removed_features_count": removed_features_count,
            "all_zero_variance": bool(np.all(variances == 0.0)),
        }

        # Store result with full report DataFrame for visualization tools
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: Union[int, float] = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates Pearson correlation (signed r) between each term's frequency in session.feature_matrix
        and a binary indicator vector for target_class.

        Filtering rule: Terms with pearson_r >= threshold and non-constant term values (std > 0) are kept.
        Constant terms (std == 0) are assigned r = 0.0 and explicitly excluded from kept features even if threshold <= 0.

        Args:
            target_class: Target category label to calculate correlation against (e.g. 'catB').
            threshold: Minimum Pearson r correlation threshold. Features with r >= threshold are kept. Default is 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_features,
            kept_features_count, and removed_features_count.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be initialized first using build_dtm_tool.")

        X = session.feature_matrix
        n_samples = X.shape[0]

        # Determine target category labels
        labels = None
        if session.labels is not None:
            labels = np.array(session.labels)
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None or len(labels) == 0:
            raise ValueError("Document labels are missing in session. Load a dataset first.")

        # Check that labels length matches DTM rows
        if len(labels) != n_samples:
            raise ValueError(f"Mismatch between labels count ({len(labels)}) and DTM rows ({n_samples}).")

        # Validate target_class existence
        unique_labels = np.unique(labels)
        if target_class not in unique_labels:
            raise ValueError(f"target_class '{target_class}' not found in dataset labels: {list(unique_labels)}")

        # Create binary indicator vector for target_class (1 if doc in target_class, else 0)
        y = (labels == target_class).astype(float)

        # Check if target vector is constant (e.g., all 1s or all 0s)
        if np.std(y) == 0:
            raise ValueError(f"Target binary vector for '{target_class}' has zero variance.")

        feature_names = np.array(session.feature_names)
        correlations = np.zeros(X.shape[1], dtype=float)
        is_constant = np.zeros(X.shape[1], dtype=bool)

        # Convert to CSC format for efficient column slicing if sparse
        X_csc = X.tocsc() if hasattr(X, "tocsc") else X

        # Compute Pearson r for each feature column
        for j in range(X.shape[1]):
            x_col = np.array(X_csc[:, j].todense()).ravel() if hasattr(X_csc, "todense") else X_csc[:, j]
            if np.std(x_col) == 0:
                correlations[j] = 0.0
                is_constant[j] = True
            else:
                r, _ = pearsonr(x_col, y)
                correlations[j] = 0.0 if np.isnan(r) else r

        # Filter features: pearson_r >= threshold AND non-constant
        kept_mask = (correlations >= threshold) & (~is_constant)
        kept_features_count = int(np.sum(kept_mask))
        removed_features_count = len(feature_names) - kept_features_count

        # Full report DataFrame: 1st column "feature", 2nd column "pearson_r", sorted descending by pearson_r
        report_df = pd.DataFrame({
            "feature": feature_names,
            "pearson_r": correlations,
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": float(threshold),
            "total_features": len(feature_names),
            "kept_features_count": kept_features_count,
            "removed_features_count": removed_features_count,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: Union[int, float] = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates Spearman rank correlation (signed r) between each term's frequency in session.feature_matrix
        and a binary indicator vector for target_class.

        Filtering rule: Terms with spearman_r >= threshold and non-constant term values (std > 0) are kept.
        Constant terms (std == 0) are assigned r = 0.0 and explicitly excluded from kept features even if threshold <= 0.

        Args:
            target_class: Target category label to calculate correlation against (e.g. 'catB').
            threshold: Minimum Spearman r correlation threshold. Features with r >= threshold are kept. Default is 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_features,
            kept_features_count, and removed_features_count.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be initialized first using build_dtm_tool.")

        X = session.feature_matrix
        n_samples = X.shape[0]

        # Determine target category labels
        labels = None
        if session.labels is not None:
            labels = np.array(session.labels)
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None or len(labels) == 0:
            raise ValueError("Document labels are missing in session. Load a dataset first.")

        # Check that labels length matches DTM rows
        if len(labels) != n_samples:
            raise ValueError(f"Mismatch between labels count ({len(labels)}) and DTM rows ({n_samples}).")

        # Validate target_class existence
        unique_labels = np.unique(labels)
        if target_class not in unique_labels:
            raise ValueError(f"target_class '{target_class}' not found in dataset labels: {list(unique_labels)}")

        # Create binary indicator vector for target_class (1 if doc in target_class, else 0)
        y = (labels == target_class).astype(float)

        # Check if target vector is constant (e.g., all 1s or all 0s)
        if np.std(y) == 0:
            raise ValueError(f"Target binary vector for '{target_class}' has zero variance.")

        feature_names = np.array(session.feature_names)
        correlations = np.zeros(X.shape[1], dtype=float)
        is_constant = np.zeros(X.shape[1], dtype=bool)

        # Convert to CSC format for efficient column slicing if sparse
        X_csc = X.tocsc() if hasattr(X, "tocsc") else X

        # Compute Spearman r for each feature column
        for j in range(X.shape[1]):
            x_col = np.array(X_csc[:, j].todense()).ravel() if hasattr(X_csc, "todense") else X_csc[:, j]
            if np.std(x_col) == 0:
                correlations[j] = 0.0
                is_constant[j] = True
            else:
                r, _ = spearmanr(x_col, y)
                correlations[j] = 0.0 if np.isnan(r) else r

        # Filter features: spearman_r >= threshold AND non-constant
        kept_mask = (correlations >= threshold) & (~is_constant)
        kept_features_count = int(np.sum(kept_mask))
        removed_features_count = len(feature_names) - kept_features_count

        # Full report DataFrame: 1st column "feature", 2nd column "spearman_r", sorted descending by spearman_r
        report_df = pd.DataFrame({
            "feature": feature_names,
            "spearman_r": correlations,
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": float(threshold),
            "total_features": len(feature_names),
            "kept_features_count": kept_features_count,
            "removed_features_count": removed_features_count,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
