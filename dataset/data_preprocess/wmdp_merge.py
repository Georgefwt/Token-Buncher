import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd

WMDP_SCHEMA = [
    "data_source",
    "prompt",
    "ability",
    "reward_model",
]

SPLITS = ["train"]

def read_parquet_or_jsonl(file_stem: str) -> pd.DataFrame:
    """
    Given a path without extension, prioritize reading .parquet, otherwise read .jsonl
    """
    parquet_path = f"{file_stem}.parquet"
    jsonl_path = f"{file_stem}.jsonl"

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(jsonl_path):
        return pd.read_json(jsonl_path, lines=True)
    raise FileNotFoundError(f"Cannot find {parquet_path} or {jsonl_path}")

def _extract_instruction_from_prompt_obj(prompt: Any) -> str:
    """
    Extract user content from beavertails prompt field as instruction.
    - prompt is expected to be [ { "content": str, "role": "user" }, ... ] or similar
    - content may contain tokens like <|im_start|>user ... <|im_end|>
    """
    try:
        # Typically list[dict]
        prompt = prompt.tolist()
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    # First try to extract through special tokens
                    if isinstance(content, str) and content:
                        m = re.search(
                            r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>",
                            content,
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        if m:
                            return m.group(1).strip()
                        # Fallback: if no tokens, use content directly as instruction
                        return str(content).strip()
        # Fallback: if not list, convert to string
        else:
            # convert to list
            return prompt
        return ("" if prompt is None else str(prompt)).strip()
    except Exception:
        return ""  # fallback

def map_beavertails_to_wmdp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map beavertails data to WMDP_SCHEMA and remove irrelevant fields
    """
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt = row.get("prompt", None)
        instruction = _extract_instruction_from_prompt_obj(prompt)

        mapped = {
            "data_source": row.get("data_source", "BeaverTails"),
            "prompt": prompt,
            "ability": row.get("ability", "nlp"),
        }
        rows.append(mapped)

    out = pd.DataFrame(rows)
    # Ensure all columns are included
    for col in WMDP_SCHEMA:
        if col not in out.columns:
            out[col] = None
    # Keep only required columns and arrange in order
    out = out[WMDP_SCHEMA]
    return out

def ensure_schema(df: pd.DataFrame, default_data_source: Optional[str]=None) -> pd.DataFrame:
    """
    Light validation/completion for wmdp data, unify column order and missing default values
    """
    out = df.copy()
    # Fill in missing columns
    if "instruction" not in out.columns:
        out["instruction"] = ""
    if "response" not in out.columns:
        out["response"] = ""
    if "data_source" not in out.columns:
        out["data_source"] = default_data_source or "wmdp"
    if "prompt" not in out.columns:
        out["prompt"] = None
    if "ability" not in out.columns:
        out["ability"] = "nlp"
    if "reward_model" not in out.columns:
        out["reward_model"] = None
    if "extra_info" not in out.columns:
        out["extra_info"] = None

    # Keep only fields in schema and arrange in order
    out = out[WMDP_SCHEMA]
    return out

def to_jsonl(df: pd.DataFrame, path: str):
    """
    When writing JSONL, ensure NaN -> None and Chinese characters are not escaped
    """
    # Set top-level NaN to None, preserve nested objects
    df = df.where(pd.notna(df), None)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

def save_split(df: pd.DataFrame, out_dir: str, split: str):
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{split}.parquet")
    jsonl_path = os.path.join(out_dir, f"{split}.jsonl")
    # Write parquet
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    # Then write jsonl
    to_jsonl(df, jsonl_path)

def _repeat_or_sample_to_n(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Expand/sample df to exactly n rows.
    - If len(df) >= n: randomly sample n rows (reproducible)
    - If len(df) < n: repeat entire table several times, then sample remaining rows without replacement
    """
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty, cannot expand to specified number of rows.")
    if len(df) >= n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    q, r = divmod(n, len(df))
    parts = [df] * q
    if r > 0:
        parts.append(df.sample(n=r, random_state=seed, replace=False))
    out = pd.concat(parts, ignore_index=True)
    # Shuffle for safety
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def _to_json_str_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    # Other types (str/int/bool etc.) unified as strings to avoid mixing with struct
    return str(v)

def normalize_nested_columns_for_parquet(df: pd.DataFrame, cols=("prompt", "extra_info", "reward_model")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(_to_json_str_or_none)
    return out


def main(wmdp_path: str, beavertails_path: str, outdir_path: str, bt_n: int, wmdp_n: int, seed: int):
    # ===== for train split =====
    # Read wmdp_path
    wmdp_df = read_parquet_or_jsonl(os.path.join(wmdp_path, "train"))
    wmdp_df = ensure_schema(wmdp_df, default_data_source="wmdp")

    # Read beavertails
    bt_df = read_parquet_or_jsonl(os.path.join(beavertails_path, "train"))
    bt_df = map_beavertails_to_wmdp(bt_df)

    # Sample & expand
    bt_sampled = _repeat_or_sample_to_n(bt_df, bt_n, seed=seed)     # Extract bt_n rows from beavertails
    bt_sampled["reward_model"] = [{"ground_truth":{"response":"EM"},"style":"rule"}] * len(bt_sampled)
    wmdp_upsampled = _repeat_or_sample_to_n(wmdp_df, wmdp_n, seed=seed)  # Expand wmdpbio to wmdp_n rows

    # Merge and shuffle
    merged = pd.concat([bt_sampled, wmdp_upsampled], ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    save_split(merged, outdir_path, "train")

    # ===== for test split =====
    # just directly copy from wmdp_path
    wmdp_df_test = read_parquet_or_jsonl(os.path.join(wmdp_path, "test"))
    wmdp_df_test = ensure_schema(wmdp_df_test, default_data_source="wmdp")

    # Save
    save_split(wmdp_df_test, outdir_path, "test")

    print(f"Completed! Output saved to: {outdir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmdp_path", default="dataset/wmdpbio-qwen", help="Dataset root directory, default 'dataset'")
    parser.add_argument("--beavertails_path", default="dataset/beavertails-qwen", help="Dataset root directory, default 'dataset'")
    parser.add_argument("--outdir_path", default="dataset/beavertails-wmdpbio-qwen", help="Dataset root directory, default 'dataset'")
    parser.add_argument("--bt_n", type=int, default=1000, help="Number of rows to extract from beavertails-qwen")
    parser.add_argument("--wmdp_n", type=int, default=2000, help="Number of rows for wmdpbio-qwen through repetition/expansion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42")
    args = parser.parse_args()
    main(args.wmdp_path, args.beavertails_path, args.outdir_path, args.bt_n, args.wmdp_n, args.seed)
