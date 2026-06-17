import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import numpy as np
import pandas as pd


def proc(ridx, triu, corr, absr, mean):
    """"""
    results = []
    for cidx in range(ridx + 1, len(triu)):
        drop = (
            ""
            if corr.iloc[ridx, cidx] <= absr
            else (
                corr.columns[ridx]
                if mean.iloc[ridx] > mean.iloc[cidx]
                else corr.columns[cidx]
            )
        )

        results.append(
            [
                corr.index[ridx],
                triu.columns[cidx],
                mean.iloc[ridx],
                mean.iloc[cidx],
                triu.iloc[ridx, cidx],
                drop,
            ]
        )

    return results


def find(df, method, absr, thread):
    """"""
    corr = df.corr(method=method).abs()
    mean = corr.mean(axis=1)
    triu = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    with ThreadPoolExecutor(
        max_workers=psutil.cpu_count(logical=thread) - 1
    ) as executor:
        futures, results = [
            executor.submit(proc, ridx, triu, corr, absr, mean)
            for ridx in range(len(triu) - 1)
        ], []
        for future in as_completed(futures):
            results += future.result()

        ct = pd.DataFrame(
            results,
            columns=[
                "v1",
                "v2",
                "v1.mean",
                "v2.mean",
                "corr",
                "drop",
            ],
        )
        ct.sort_values("corr", ascending=False, inplace=True)

    return ct


def drop(ct):
    """"""
    return set(ct[ct["drop"] != ""]["drop"])


def timeit(st):
    """"""
    ed = time.time() - st
    return time.strftime("%H:%M:%S", time.gmtime(ed))


def main(df, method="pearson", absr=0.9, thread=False):
    """"""
    if (
        not isinstance(df, pd.DataFrame)
        or method not in {"pearson", "kendall", "spearman"}
        or not isinstance(absr, float)
        or absr < 0
        or not isinstance(thread, bool)
    ):
        raise ValueError(
            "Invalid parameters: "
            "'df' must be a Pandas DataFrame; "
            "'method' must be one of 'pearson', 'kendall', or 'spearman'; "
            "'absr' must be a non-negative float; "
            "'thread' must be a boolean (True or False)."
        )

    st = time.time()

    ct = find(df, method, absr, thread)
    cols = drop(ct)

    print(f"⌛ Time elapsed: {timeit(st)}")

    return cols
