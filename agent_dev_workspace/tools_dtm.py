from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) using CountVectorizer from session.dataframe["text"].

        Args:
            ngram_range: Lower and upper boundary of range of n-values for different n-grams. Default is [1, 1].
            min_df: Minimum document frequency threshold. Integer for document count, float for proportion. Default is 1.
            max_df: Maximum document frequency cutoff as a proportion in (0, 1]. Terms appearing in more than this fraction of documents are ignored. Default is 1.0.

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        # Validate ngram_range
        if not isinstance(ngram_range, list) or len(ngram_range) != 2:
            raise ValueError("ngram_range must be a list containing exactly two positive integers.")

        min_n, max_n = ngram_range[0], ngram_range[1]
        if not (isinstance(min_n, int) and isinstance(max_n, int) and min_n > 0 and max_n > 0):
            raise ValueError("ngram_range elements must both be positive integers.")

        if min_n > max_n:
            raise ValueError(f"Invalid ngram_range {ngram_range}: lower bound ({min_n}) cannot be greater than upper bound ({max_n}).")

        texts = session.dataframe["text"].fillna("")

        ngram_tuple = (min_n, max_n)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
        )

        dtm = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Sparsity calculation
        n_rows, n_cols = dtm.shape
        total_elements = n_rows * n_cols
        non_zero = int(dtm.nnz)
        sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Aggregates total term frequency counts across all documents in session.feature_matrix.

        Args:
            top_n: Number of top-frequent terms to include in top_terms summary. Default is 20.

        Returns:
            Dict containing result_id, total_terms, total_word_count, and top_terms mapping.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be initialized first using build_dtm_tool.")

        X = session.feature_matrix
        if X.shape[0] == 0:
            raise ValueError("session.feature_matrix contains 0 documents.")

        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        feature_names = np.array(session.feature_names)

        # Sum term frequencies across all documents (axis 0)
        counts = np.array(X.sum(axis=0)).ravel()
        total_word_count = int(np.sum(counts))
        total_terms = len(feature_names)

        # Construct full report DataFrame: feature as 1st column, frequency as 2nd column
        # Sort primary by frequency descending, secondary by feature name ascending for consistent tie-breaking
        report_df = pd.DataFrame({
            "feature": feature_names,
            "frequency": counts,
        }).sort_values(by=["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)

        # Take top_n terms for summary dictionary
        top_df = report_df.head(top_n)
        top_terms = dict(zip(top_df["feature"], top_df["frequency"].astype(int)))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": total_terms,
            "total_word_count": total_word_count,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=report_df)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """
        Renders a heatmap visualization of a positional slice of the raw Document-Term Matrix (DTM)
        from session.feature_matrix.

        Args:
            n_terms: Maximum number of terms (columns) to include in slice. Default is 20.
            n_documents: Maximum number of documents (rows) to include in slice. Default is 20.

        Returns:
            Dict containing result_id, n_terms, n_documents, terms, doc_labels, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be initialized first using build_dtm_tool.")

        if not isinstance(n_terms, int) or n_terms <= 0:
            raise ValueError("n_terms must be a positive integer.")
        if not isinstance(n_documents, int) or n_documents <= 0:
            raise ValueError("n_documents must be a positive integer.")

        X = session.feature_matrix
        total_docs, total_terms_count = X.shape

        if total_docs == 0 or total_terms_count == 0:
            raise ValueError("session.feature_matrix is empty.")

        # Slice sub-matrix [0:n_documents, 0:n_terms]
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms_count)

        X_slice = X[0:actual_docs, 0:actual_terms]
        dense_slice = X_slice.toarray() if hasattr(X_slice, "toarray") else np.array(X_slice)

        sliced_terms = session.feature_names[0:actual_terms]
        doc_labels = [f"Doc {i}" for i in range(actual_docs)]

        # Generate heatmap plot using seaborn and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"Document-Term Matrix Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending_figure on session without calling plt.savefig or plt.show
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": actual_terms,
            "n_documents": actual_docs,
            "terms": sliced_terms,
            "doc_labels": doc_labels,
            "matrix_slice": dense_slice.tolist(),
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
