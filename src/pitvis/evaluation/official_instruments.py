"""VENDORED — the official PitVis instrument metric. Do not edit.

Copied verbatim from the challenge organisers' repository:

    https://github.com/dreets/pitvis
    helper_scripts/evaluation_instruments.py
    commit ebc82dde642f0abd4da1baf2fa623bfd788a4fa6 (2023-09-11)
    sha256 e906f30135edbdace3c06bb855c900c47765d4d5939a15bcfa03e133bb918c8b

(Note the default branch is `trunk`, not `main`.)

The ONLY modification is this docstring. Everything below is unmodified, so the
file stays diffable against upstream and the headline number is the challenge's
number by construction. Naming style (`ls_`/`flt_`/`int_`/`df_` prefixes) is
theirs; do not "fix" it.

Consumed by `pitvis.evaluation.instruments`. This is TASK 2 and differs from the
steps metric (`official.py`) in four ways that all matter:

- `average="weighted"`, NOT macro. The paper's §3.4.3 and Table 6's own column
  header both say macro-F1; the shipped code computes weighted. We follow the
  code, on the same precedent as the Eq-3 edit-score conflict — but we print
  both numbers, because which one produced the published 41.6 is unknowable.
- `remove_background_insts` is defined but its call is COMMENTED OUT in
  `clean_insts`. Out-of-patient frames therefore survive as all-zero rows and
  are scored. The steps metric does the opposite: it drops those rows.
- There is no edit score. Instruments are multi-label, so a single sequence
  cannot be collapsed by `groupby` the way steps can.
- No classes are excluded on rarity grounds (steps drop 11 and 13).

`hot_encode_insts` keeps 19 columns, ids 0..18, after popping -1 and -2. Class 0
("no visible instrument") is therefore a SCORED class, not a sentinel — it holds
31.5% of frames and dominates a support-weighted average.

Known upstream defect, preserved deliberately: `hot_encode_insts` fits a
SEPARATE MultiLabelBinarizer on trues and on preds, then appends whichever
columns are missing. When the two observe different class sets the resulting
column ORDERS differ, and `f1_score` compares DataFrames positionally. See
`pitvis.evaluation.instruments` for how we detect this rather than inherit it
silently.
"""

# global imports
import pandas as pd
from sklearn import metrics

# strongly typed
from typing import List
from typing import Tuple
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    # ls_trues: List[List] = []  # ground-truth instruments per frame (2 integer values)
    # ls_preds: List[List] = []  # prediction instruments per frame (1 or 2 integer values)
    # flt_evaluation_metric: float = calculate_insts_evaluation_metric(ls_trues=ls_trues, ls_preds=ls_preds)
    pass


def calculate_insts_evaluation_metric(ls_trues: List[List], ls_preds: List[List]) -> float:
    """Calculate the instrument evaluation metric from a ground-truth list and prediction list."""
    df_trues_encoded, df_preds_encoded = clean_insts(ls_trues=ls_trues, ls_preds=ls_preds)
    flt_f1_score: float = metrics.f1_score(
        y_true=df_trues_encoded,
        y_pred=df_preds_encoded,
        average="weighted",
        zero_division=1,
    )
    return flt_f1_score


def clean_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[DataFrame, DataFrame]:
    """Ensure input data is compatible with evaluation metric calculation."""
    ls_trues_pad, ls_preds_pad = check_insts_lists_are_compatible(ls_trues=ls_trues, ls_preds=ls_preds)
    # ls_trues_no_bkg, ls_preds_no_bkg = remove_background_insts(ls_trues=ls_trues_pad, ls_preds=ls_preds_pad)
    df_trues_enc, df_preds_enc = hot_encode_insts(ls_trues=ls_trues_pad, ls_preds=ls_preds_pad)
    return df_trues_enc, df_preds_enc


def check_insts_lists_are_compatible(ls_trues: List[List], ls_preds: List[List]) -> Tuple[List[List], List[List]]:
    """Ensure truths and predictions are compatible, pad with (-1, -2) when necessary."""
    if len(ls_trues) != len(ls_preds):
        print(f"Lengths of truths ({len(ls_trues)}) and preds ({len(ls_preds)}) are not equal.")
        raise SystemExit
    for int_index, ls_true in enumerate(ls_trues):
        if len(ls_true) != 2:
            print(f"Lengths of truths at index={int_index} is {len(ls_true)}!=2.")
            raise SystemExit

    ls_preds_padded: List[List] = []
    for int_index, ls_pred in enumerate(ls_preds):
        if len(ls_pred) == 0:
            ls_pred_padded: List = [-1, -2]
        elif len(ls_pred) == 1:
            ls_pred_padded: List = [ls_pred[0], -2]
        elif len(ls_pred) == 2:
            ls_pred_padded: List = ls_pred
        else:
            print(f"Lengths of truths at index={int_index} is {len(ls_pred)}>2.")
            raise SystemExit
        ls_preds_padded.append(ls_pred_padded)
    return ls_trues, ls_preds_padded


def remove_background_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[List[List], List[List]]:
    """Remove background class (-1), as defined by the the ground-truth."""
    df_trues: DataFrame = pd.DataFrame(ls_trues, columns=["inst1", "inst2"])
    df_preds: DataFrame = pd.DataFrame(ls_preds, columns=["inst3", "inst4"])
    df_trues_preds: DataFrame = pd.concat([df_trues, df_preds], axis=1)

    # check which column contains the "out_of_patient" class -1 (usually 'inst1') and remove those frames
    int_background_inst1: int = df_trues_preds[df_trues_preds["inst1"] == -1]["inst1"].count()
    int_background_inst2: int = df_trues_preds[df_trues_preds["inst2"] == -1]["inst2"].count()
    if int_background_inst1 > int_background_inst2:
        df_trues_preds_no_background: DataFrame = df_trues_preds[df_trues_preds["inst1"] != -1]
    else:
        df_trues_preds_no_background: DataFrame = df_trues_preds[df_trues_preds["inst2"] != -1]

    ls_trues_no_background: List[List] = df_trues_preds_no_background[["inst1", "inst2"]].to_numpy().tolist()
    ls_preds_no_background: List[List] = df_trues_preds_no_background[["inst3", "inst4"]].to_numpy().tolist()
    return ls_trues_no_background, ls_preds_no_background


def hot_encode_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[DataFrame, DataFrame]:
    """Hot encode the both ground-truth and predictions."""
    mlb = MultiLabelBinarizer()
    df_trues = pd.Series(ls_trues)
    df_preds = pd.Series(ls_preds)
    df_trues_encoded: DataFrame = pd.DataFrame(mlb.fit_transform(df_trues), columns=mlb.classes_, index=df_trues.index)
    df_preds_encoded: DataFrame = pd.DataFrame(mlb.fit_transform(df_preds), columns=mlb.classes_, index=df_preds.index)

    # replacing blanks with 0
    ls_range: List[int] = [int_x for int_x in range(-2, 19)]
    for int_inst in ls_range:
        if int_inst not in df_trues_encoded.columns.to_list():
            df_trues_encoded[int_inst] = [0] * len(df_trues_encoded)
    for int_inst in ls_range:
        if int_inst not in df_preds_encoded.columns.to_list():
            df_preds_encoded[int_inst] = [0] * len(df_trues_encoded)

    # removing background classes
    df_trues_encoded.pop(-1)
    df_preds_encoded.pop(-1)
    df_trues_encoded.pop(-2)
    df_preds_encoded.pop(-2)
    return df_trues_encoded, df_preds_encoded


if __name__ == "__main__":
    main()
