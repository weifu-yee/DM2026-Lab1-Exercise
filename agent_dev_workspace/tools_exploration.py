from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_idx1: int, doc_idx2: int) -> Dict[str, Any]:
        """Computes cosine similarity between two documents using session.artifacts["count_vectorizer"].

        Transforms text at doc_idx1 and doc_idx2 using the stored vectorizer (without refitting),
        calculating dot product over vector norms. Zero-vector documents yield a similarity of 0.0.

        Args:
            doc_idx1: Integer row index of the first document in session.dataframe.
            doc_idx2: Integer row index of the second document in session.dataframe.

        Returns:
            Dict containing result_id, doc_idx1, doc_idx2, and calculated cosine similarity score.

        Raises:
            ValueError: If session.dataframe or session.artifacts['count_vectorizer'] is missing,
                       text column is missing, or indices are out of bounds.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe found in session. Call load_dataset_tool first.")

        if "count_vectorizer" not in session.artifacts or session.artifacts["count_vectorizer"] is None:
            raise ValueError(
                "No 'count_vectorizer' found in session.artifacts. Call build_dtm_tool first."
            )

        vectorizer = session.artifacts["count_vectorizer"]
        n_docs = len(session.dataframe)

        if not (0 <= doc_idx1 < n_docs):
            raise ValueError(f"doc_idx1={doc_idx1} is out of bounds for dataset with {n_docs} documents.")

        if not (0 <= doc_idx2 < n_docs):
            raise ValueError(f"doc_idx2={doc_idx2} is out of bounds for dataset with {n_docs} documents.")

        text_col = "text" if "text" in session.dataframe.columns else None
        if not text_col:
            raise ValueError("No 'text' column found in session.dataframe.")

        val1 = session.dataframe.iloc[doc_idx1][text_col]
        val2 = session.dataframe.iloc[doc_idx2][text_col]

        text1 = "" if pd.isna(val1) else str(val1)
        text2 = "" if pd.isna(val2) else str(val2)

        # Transform selected document text using existing count_vectorizer (no fitting)
        v1 = vectorizer.transform([text1])
        v2 = vectorizer.transform([text2])

        v1_arr = v1.toarray().ravel()
        v2_arr = v2.toarray().ravel()

        norm1 = np.linalg.norm(v1_arr)
        norm2 = np.linalg.norm(v2_arr)

        # Handle empty documents / zero vectors cleanly
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            dot_product = np.dot(v1_arr, v2_arr)
            similarity = float(dot_product / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": similarity,
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_idx1": doc_idx1, "doc_idx2": doc_idx2},
            summary=summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes Pearson correlation matrix over top-N terms by variance from global DTM.

        Renders a annotated correlation heatmap stored in session.pending_figure and returns
        the selected term names and correlation matrix.

        Args:
            top_n: Number of top-variance terms to select (default 20).

        Returns:
            Dict containing result_id, terms, and correlation_matrix.

        Raises:
            ValueError: If session.feature_matrix or feature_names is missing, top_n <= 0,
                       or any selected term has zero variance (constant column).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No feature_matrix or feature_names in session. Call build_dtm_tool first."
            )

        if top_n <= 0:
            raise ValueError(f"top_n must be a positive integer, got {top_n}.")

        dtm = session.feature_matrix
        feature_names = session.feature_names
        n_docs, n_terms = dtm.shape

        if n_docs < 2:
            raise ValueError(f"Insufficient documents for correlation ({n_docs} found, need >= 2).")

        if n_terms == 0:
            raise ValueError("Feature matrix has 0 terms.")

        # Efficient sparse variance computation across columns without whole-matrix densification
        if issparse(dtm):
            # E[X^2] - (E[X])^2
            mean = np.ravel(dtm.mean(axis=0))
            sq_mean = np.ravel(dtm.power(2).mean(axis=0))
            variances = sq_mean - (mean ** 2)
        else:
            variances = np.var(dtm, axis=0)

        # Select top-N terms by variance
        actual_k = min(top_n, n_terms)
        top_indices = np.argsort(variances)[::-1][:actual_k]

        selected_terms = [feature_names[i] for i in top_indices]
        selected_variances = variances[top_indices]

        # Check if any selected term has zero variance
        zero_var_terms = [selected_terms[i] for i, v in enumerate(selected_variances) if v <= 0]
        if zero_var_terms:
            raise ValueError(
                f"Cannot compute correlation: term(s) {zero_var_terms} have zero variance (constant column)."
            )

        # Densify ONLY the selected top-N columns
        if issparse(dtm):
            dense_cols = dtm[:, top_indices].toarray()
        else:
            dense_cols = dtm[:, top_indices]

        # Compute Pearson correlation matrix across selected terms
        # rowvar=False means columns are variables
        corr_matrix = np.corrcoef(dense_cols, rowvar=False)

        # Handle case when top_n=1 (returns scalar from np.corrcoef)
        if actual_k == 1:
            corr_matrix = np.array([[1.0]])

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_k * 0.8), max(5, actual_k * 0.7)))
        sns.heatmap(
            corr_matrix,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top-{actual_k} Terms by Variance)")
        plt.tight_layout()

        # Set pending figure for session display/save
        session.pending_figure = fig

        corr_df = pd.DataFrame(corr_matrix, index=selected_terms, columns=selected_terms)
        corr_list = np.round(corr_matrix, 4).tolist()

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=corr_df,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
