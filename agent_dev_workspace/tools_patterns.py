from typing import Dict, Any, List, Optional, Literal
import tempfile
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.maximal import MaxFPGrowth
from PAMI.frequentPattern.topk import FAE


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent patterns for a specific category using PAMI algorithms.

        Fits a category-local CountVectorizer(stop_words="english"), applies vocabulary filtering
        (variance, tfidf, or term_frequency), and mines patterns using fpgrowth, maxfpgrowth, or topk.

        Args:
            category_name: Exact category string to filter dataframe rows by.
            filtering_method: Vocabulary filter ("variance", "tfidf", "term_frequency"). Default "tfidf".
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", "maxfpgrowth"). Default "fpgrowth".
            min_sup: Minimum support threshold (required for fpgrowth and maxfpgrowth).
            k: Number of top patterns to return (required for topk).

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, and patterns list.

        Raises:
            ValueError: If dataframe is missing, category_name not found, or required parameters
                       (min_sup for fpgrowth/maxfpgrowth, k for topk) are missing/invalid.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe found in session. Call load_dataset_tool first.")

        # Determine category column name
        if "category_name" in session.dataframe.columns:
            cat_col = "category_name"
        elif "category" in session.dataframe.columns:
            cat_col = "category"
        else:
            raise ValueError("No 'category_name' or 'category' column found in session.dataframe.")

        # Filter dataframe by category_name
        cat_df = session.dataframe[session.dataframe[cat_col].astype(str) == str(category_name)]
        if cat_df.empty:
            raise ValueError(f"Category '{category_name}' not found or has 0 documents.")

        # Validate algorithm parameters
        alg_lower = algorithm.lower()
        if alg_lower in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                raise ValueError(f"min_sup must be a positive integer for algorithm '{algorithm}'.")
        elif alg_lower == "topk":
            if k is None or k <= 0:
                raise ValueError("k must be a positive integer for algorithm 'topk'.")
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'fpgrowth', 'maxfpgrowth', or 'topk'.")

        text_col = "text" if "text" in cat_df.columns else None
        if not text_col:
            raise ValueError("No 'text' column found in session.dataframe.")

        # Step 1: Category-local DTM with English stop words
        texts = ["" if pd.isna(t) else str(t) for t in cat_df[text_col]]
        vectorizer = CountVectorizer(stop_words="english")
        
        filter_lower = filtering_method.lower()
        artifact_key = f"patterns_{category_name}_{filter_lower}_{alg_lower}"

        try:
            dtm_cat = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Handle empty vocabulary (e.g. all words removed by stop words)
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filter_lower,
                "algorithm": alg_lower,
                "patterns": [],
                "note": "empty vocabulary after English stop words removal",
            }
            session.store_result(
                "mine_patterns_tool",
                args={"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm},
                summary=summary,
                full_report=pd.DataFrame(columns=["pattern", "support"])
            )
            return summary

        if dtm_cat.shape[1] == 0 or len(feature_names) == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filter_lower,
                "algorithm": alg_lower,
                "patterns": [],
                "note": "no terms survived English stop words removal",
            }
            session.store_result(
                "mine_patterns_tool",
                args={"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm},
                summary=summary,
                full_report=pd.DataFrame(columns=["pattern", "support"])
            )
            return summary

        # Step 2: Vocabulary filtering (inclusive 5th-95th percentile bounds like Master)
        dense_dtm = dtm_cat.toarray()
        kept_indices = []

        if filter_lower == "variance":
            variances = np.var(dense_dtm, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            kept_indices = np.where((variances >= p5) & (variances <= p95))[0]

        elif filter_lower == "term_frequency":
            counts = np.sum(dense_dtm, axis=0)
            p5 = np.percentile(counts, 5)
            p95 = np.percentile(counts, 95)
            kept_indices = np.where((counts >= p5) & (counts <= p95))[0]

        elif filter_lower == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dense_dtm).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_indices = np.where(mean_tfidf >= p20)[0]

        else:
            raise ValueError(f"Unknown filtering_method '{filtering_method}'. Choose 'variance', 'tfidf', or 'term_frequency'.")

        # Handle zero terms surviving filtering
        if len(kept_indices) == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filter_lower,
                "algorithm": alg_lower,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result(
                "mine_patterns_tool",
                args={
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary=summary,
                full_report=pd.DataFrame(columns=["pattern", "support"])
            )
            return summary

        # Step 3: Map kept terms to ASCII item IDs (item_0, item_1, ...) to avoid encoding mismatches
        # in PAMI DF2DB's open() calls on Windows (default cp950/system encoding vs UTF-8 PAMI readers).
        filtered_terms = feature_names[kept_indices]
        filtered_counts = dense_dtm[:, kept_indices]

        ascii_columns = [f"item_{i}" for i in range(len(filtered_terms))]
        item_id_to_term = dict(zip(ascii_columns, filtered_terms))

        filtered_df = pd.DataFrame(filtered_counts, columns=ascii_columns)

        # Step 4: Convert to transactional database file via PAMI DF2DB
        temp_db_file = None
        patterns_list = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
                temp_db_file = tmp.name

            # PAMI DF2DB conversion
            conv = db_converter.DF2DB(filtered_df)
            conv.convert2TransactionalDatabase(temp_db_file, ">=", 1)

            # Step 5: Mine patterns using chosen PAMI algorithm
            if alg_lower == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=temp_db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                pami_df = miner.getPatternsAsDataFrame()

            elif alg_lower == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=temp_db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                pami_df = miner.getPatternsAsDataFrame()

            elif alg_lower == "topk":
                miner = FAE.FAE(iFile=temp_db_file, k=k, sep="\t")
                miner.startMine()
                pami_df = miner.getPatternsAsDataFrame()

            # Parse mined patterns from PAMI DataFrame and map ASCII item IDs back to original terms
            if pami_df is not None and not pami_df.empty:
                for _, row in pami_df.iterrows():
                    pattern_str = str(row["Patterns"]).strip()
                    support_val = int(row["Support"])
                    raw_items = [item.strip() for item in pattern_str.split() if item.strip()]
                    mapped_items = [item_id_to_term.get(item, item) for item in raw_items]
                    patterns_list.append({"pattern": mapped_items, "support": support_val})

        finally:
            if temp_db_file and os.path.exists(temp_db_file):
                try:
                    os.remove(temp_db_file)
                except Exception:
                    pass

        # Store full pattern list in session artifacts
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filter_lower,
            "algorithm": alg_lower,
            "patterns": patterns_list[:20],  # Truncate summary to top 20
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary=summary,
            full_report=pd.DataFrame(patterns_list) if patterns_list else pd.DataFrame(columns=["pattern", "support"]),
        )

        return summary

    return [mine_patterns_tool]
