from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels into binary indicator columns for each class.

        Validates presence of data and labels, synchronizes labels via session.set_labels,
        and creates one binary column per unique class.

        Returns:
            Dict containing summary of binarization (result_id, classes, shape, column_names).
        
        Raises:
            ValueError: If dataframe/category column is missing, dataframe is empty,
                       or any label is missing/blank.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        if session.dataframe.empty:
            raise ValueError("Session dataframe is empty.")

        # Check for category label column
        if "category_name" in session.dataframe.columns:
            labels_series = session.dataframe["category_name"]
        elif "category" in session.dataframe.columns:
            labels_series = session.dataframe["category"]
        else:
            raise ValueError("No 'category_name' or 'category' column found in session.dataframe.")

        # Check for missing or blank labels
        if labels_series.isna().any():
            raise ValueError("Dataset contains missing (NaN) category labels.")

        labels_str = labels_series.astype(str)
        if (labels_str.str.strip() == "").any():
            raise ValueError("Dataset contains blank/empty string category labels.")

        # Synchronize labels and categories on session
        session.set_labels(labels_str.tolist())

        # One-hot encoding with one column per unique class
        one_hot_df = pd.get_dummies(labels_str, prefix="", prefix_sep="").astype(int)
        
        class_names = [str(c) for c in one_hot_df.columns]
        one_hot_df.columns = class_names

        # Store in artifacts
        session.artifacts["binarized_labels"] = one_hot_df

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": class_names,
            "shape": list(one_hot_df.shape),
            "column_names": class_names,
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=one_hot_df
        )

        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces dimensionality of session.feature_matrix using PCA, t-SNE, or UMAP to 2D.

        Generates a 2D scatter plot colored by category on session.pending_figure. Stores all
        reduced coordinates in session.artifacts['reduced_coordinates'] and full_report.

        Args:
            method: Reduction algorithm ("pca", "tsne", or "umap"). Default "pca".
            n_components: Number of target dimensions (must be 2). Default 2.
            perplexity: Perplexity parameter for t-SNE (must be < n_samples). Default 30.0.
            n_neighbors: Number of neighbors for UMAP (must be >= 2 and < n_samples). Default 15.
            random_state: Seed for reproducibility.

        Returns:
            Dict summary with result_id, method, n_components, truncated coordinates (top 20),
            and explained_variance_ratio (for PCA).

        Raises:
            ValueError: If session.feature_matrix or session.dataframe is missing, n_components != 2,
                       category labels are missing/unaligned, or parameters are out of bounds.
        """
        if session.feature_matrix is None:
            raise ValueError("No feature_matrix found in session. Call build_dtm_tool first.")

        if session.dataframe is None:
            raise ValueError("No dataframe found in session. Call load_dataset_tool first.")

        if n_components != 2:
            raise ValueError(f"n_components must be exactly 2 for visualization, got {n_components}.")

        n_samples = session.feature_matrix.shape[0]
        if n_samples == 0:
            raise ValueError("Feature matrix has 0 documents.")

        if len(session.dataframe) != n_samples:
            raise ValueError(
                f"Mismatch: dataframe has {len(session.dataframe)} rows, but feature_matrix has {n_samples} rows."
            )

        # Check category column existence
        if "category_name" in session.dataframe.columns:
            cat_series = session.dataframe["category_name"]
        elif "category" in session.dataframe.columns:
            cat_series = session.dataframe["category"]
        else:
            raise ValueError("No 'category_name' or 'category' column found in session.dataframe.")

        # Extract positional category array to prevent pandas index misalignment
        categories_array = cat_series.to_numpy()

        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            model = PCA(n_components=2, random_state=random_state)
            coords = model.fit_transform(
                session.feature_matrix.toarray()
                if hasattr(session.feature_matrix, "toarray")
                else session.feature_matrix
            )
            explained_variance_ratio = np.round(model.explained_variance_ratio_, 4).tolist()

        elif method_lower == "tsne":
            if perplexity >= n_samples:
                raise ValueError(
                    f"perplexity ({perplexity}) must be less than n_samples ({n_samples})."
                )
            if perplexity <= 0:
                raise ValueError(f"perplexity must be positive, got {perplexity}.")

            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(
                session.feature_matrix.toarray()
                if hasattr(session.feature_matrix, "toarray")
                else session.feature_matrix
            )

        elif method_lower == "umap":
            if n_neighbors < 2:
                raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}.")
            if n_neighbors >= n_samples:
                raise ValueError(
                    f"n_neighbors ({n_neighbors}) must be strictly less than n_samples ({n_samples})."
                )

            model = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(
                session.feature_matrix.toarray()
                if hasattr(session.feature_matrix, "toarray")
                else session.feature_matrix
            )

        else:
            raise ValueError(f"Unknown reduction method '{method}'. Choose 'pca', 'tsne', or 'umap'.")

        coords_rounded = np.round(coords, 4)

        # Store all coordinates in artifacts
        session.artifacts["reduced_coordinates"] = coords_rounded

        # Plot 2D scatter plot colored by category using positional array
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords_rounded[:, 0],
            y=coords_rounded[:, 1],
            hue=categories_array,
            style=categories_array,
            s=80,
            ax=ax,
        )
        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        # Build full report DataFrame
        full_report_df = pd.DataFrame(
            coords_rounded,
            columns=["dim_1", "dim_2"],
            index=session.dataframe.index,
        )
        full_report_df["category"] = categories_array

        # Truncate summary coordinates to top 20 rows to keep summary small
        summary_coords = coords_rounded[:20].tolist()

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": 2,
            "n_samples": n_samples,
            "coordinates": summary_coords,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary=summary,
            full_report=full_report_df,
        )

        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
