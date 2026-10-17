import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """
        Lists files and subdirectories in a directory relative to the repository root.

        Args:
            subdirectory: Subdirectory path relative to the repository root. Default is '.'.

        Returns:
            Dict containing result_id, subdirectory, directories, and files.
        """
        # Determine actual repo root (if current directory ends with agent_dev_workspace, go to parent)
        cwd = Path.cwd().resolve()
        repo_root = cwd.parent if cwd.name == "agent_dev_workspace" else cwd

        target_path = (repo_root / subdirectory).resolve()

        if not target_path.exists():
            raise ValueError(f"Path '{subdirectory}' does not exist.")
        if not target_path.is_dir():
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        directories = []
        files = []

        for item in sorted(os.listdir(target_path)):
            if item.startswith("."):
                continue  # Skip hidden files/directories
            item_path = target_path / item
            if item_path.is_dir():
                directories.append(item)
            elif item_path.is_file():
                files.append(item)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
    ) -> Dict[str, Any]:
        """
        Peeks at the first n_rows of session.dataframe without modifying it.

        Args:
            n_rows: Number of rows to inspect. Must be a positive integer. Default is 5.

        Returns:
            Dict containing result_id, total_rows, columns, n_rows_inspected, and preview.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing or uninitialized.")
        if not isinstance(n_rows, int) or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")

        df = session.dataframe
        total_rows = len(df)
        columns = list(df.columns)

        # Slice top n_rows
        df_head = df.head(n_rows)

        # Convert records safely handling lists, scalar missing values, and numpy/pandas types
        preview = []
        for record in df_head.to_dict(orient="records"):
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, list):
                    clean_record[k] = [None if pd.isna(item) else item for item in v]
                elif pd.isna(v):
                    clean_record[k] = None
                else:
                    clean_record[k] = v
            preview.append(clean_record)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": columns,
            "n_rows_inspected": len(df_head),
            "preview": preview,
        }

        # Verify JSON serializability with allow_nan=False
        json.dumps(summary, allow_nan=False)

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes each document's text in session.dataframe into unigrams using nltk.word_tokenize.

        Returns:
            Dict containing result_id, n_documents, total_tokens, and avg_tokens_per_doc.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        unigrams_list = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not str(text).strip():
                unigrams_list.append([])
            else:
                try:
                    tokens = nltk.word_tokenize(str(text).lower())
                except LookupError as e:
                    raise LookupError(
                        "NLTK tokenizer data is missing. Please run `nltk.download('punkt')` "
                        "and `nltk.download('punkt_tab')` to install the required tokenizer models."
                    ) from e
                unigrams_list.append(tokens)

        # Update session state
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        total_tokens = sum(len(tokens) for tokens in unigrams_list)
        n_documents = len(unigrams_list)
        avg_tokens = round(total_tokens / n_documents, 4) if n_documents > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks session.dataframe for missing or empty text values in the 'text' column.

        Returns:
            Dict containing result_id, n_documents, missing_count, missing_pct, and missing_indices.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        df = session.dataframe
        missing_indices = []

        for idx, text in enumerate(df["text"]):
            if pd.isna(text) or not str(text).strip():
                missing_indices.append(idx)

        n_documents = len(df)
        missing_count = len(missing_indices)
        missing_pct = round(100.0 * missing_count / n_documents, 4) if n_documents > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False,
    ) -> Dict[str, Any]:
        """
        Checks for duplicate documents based on the 'text' column in session.dataframe.
        When drop=True, removes all copies of duplicated rows using keep=False.

        Args:
            drop: If True, drops all copies of duplicated rows in session.dataframe. Default is False.

        Returns:
            Dict containing result_id, initial_documents, duplicate_count, dropped_count, and remaining_documents.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        df = session.dataframe
        initial_documents = len(df)

        # Count duplicate extra occurrences
        duplicate_mask = df.duplicated(subset=["text"], keep="first")
        duplicate_count = int(duplicate_mask.sum())

        dropped_count = 0
        if drop:
            # Find all copies of duplicated rows using keep=False
            all_dups_mask = df.duplicated(subset=["text"], keep=False)
            dropped_count = int(all_dups_mask.sum())

            # Drop all duplicate rows and reset index
            new_df = df[~all_dups_mask].reset_index(drop=True)
            session.dataframe = new_df

            # Update session artifacts if unigrams exist in new_df
            if "unigrams" in new_df.columns:
                session.artifacts["unigrams"] = list(new_df["unigrams"])

            # Clear stale DTM, labels, and related artifacts since row count/alignment changed
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            session.categories = []
            session.artifacts.pop("count_vectorizer", None)

        remaining_documents = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "initial_documents": initial_documents,
            "duplicate_count": duplicate_count,
            "dropped_count": dropped_count,
            "remaining_documents": remaining_documents,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: int = 10,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Subsamples session.dataframe to n rows reproducibly using pandas DataFrame.sample.

        Args:
            n: Number of documents to sample. Must be positive and <= total documents. Default is 10.
            random_state: Random seed for reproducible sampling. Default is None.

        Returns:
            Dict containing result_id, initial_documents, sampled_documents, sampled_indices, and random_state.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        df = session.dataframe
        initial_documents = len(df)

        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if n > initial_documents:
            raise ValueError(f"Requested sample size n={n} exceeds total available documents ({initial_documents}).")

        # Subsample using pandas sample
        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session dataframe with reset index
        new_df = sampled_df.reset_index(drop=True)
        session.dataframe = new_df

        # Synchronize unigrams artifact if present
        if "unigrams" in new_df.columns:
            session.artifacts["unigrams"] = list(new_df["unigrams"])

        # Clear stale DTM, labels, and vectorizer artifacts
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None
        session.categories = []
        session.artifacts.pop("count_vectorizer", None)

        sampled_documents = len(new_df)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "initial_documents": initial_documents,
            "sampled_documents": sampled_documents,
            "sampled_indices": sampled_indices,
            "random_state": random_state,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes text character length statistics on session.dataframe overall and per category,
        and builds a box plot set on session.pending_figure.

        Returns:
            Dict containing result_id, n_documents, text_length_stats, and category_stats.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")
        if len(session.dataframe) == 0:
            raise ValueError("session.dataframe is empty.")

        df = session.dataframe
        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Describe overall stats
        desc = df["text_length"].describe()
        text_length_stats = {
            "count": int(desc["count"]),
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4) if pd.notna(desc["std"]) else None,
            "min": round(float(desc["min"]), 4),
            "25%": round(float(desc["25%"]), 4),
            "50%": round(float(desc["50%"]), 4),
            "75%": round(float(desc["75%"]), 4),
            "max": round(float(desc["max"]), 4),
        }

        # Describe per-category stats if category_name column exists
        category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                cat_desc = group["text_length"].describe()
                category_stats[str(cat)] = {
                    "count": int(cat_desc["count"]),
                    "mean": round(float(cat_desc["mean"]), 4),
                    "std": round(float(cat_desc["std"]), 4) if pd.notna(cat_desc["std"]) else None,
                    "min": round(float(cat_desc["min"]), 4),
                    "25%": round(float(cat_desc["25%"]), 4),
                    "50%": round(float(cat_desc["50%"]), 4),
                    "75%": round(float(cat_desc["75%"]), 4),
                    "max": round(float(cat_desc["max"]), 4),
                }

        # Plot boxplot grouped by category_name (or single boxplot if no category column)
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "text_length_stats": text_length_stats,
            "category_stats": category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        tokenize_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
