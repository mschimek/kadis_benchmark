#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.ticker as ticker
import argparse
import logparser
from pathlib import Path
import sys
import math
import re

key_value_paths = {
    "iterations": ["config", "iteration"],
    "p": ["config", "num_pe"],
    "datasize": ["config", "datasize"],
    "not_use_dma": ["config", "not_use_dma"],
    "not_use_mpi_colls": ["config", "not_use_mpi_colls"],
    "num_levels": ["config", "num_levels"],
    "hierarchy_aware": ["config", "hiearchy_aware"],
}


def parse_logs(experiment_path, experiment_names):
    value_paths = {
        "total_time": ["data", "root", "total_time", "statistics", "max"],
    }
    value_paths.update(key_value_paths)
    df = []
    for exp in experiment_names:
        experiment_path_ = Path(experiment_path) / exp
        if not experiment_path_.exists():
            print("The experiment directory {} doesn't exist".format(experiment_path_))
            sys.exit(1)
        df.append(
            logparser.read_logs_from_directory(experiment_path_, "intel", value_paths)
        )

    df = pd.concat(df)
    df = df.query("iteration > 0")
    return df


def parse_counter_logs(experiment_path, experiment_names):
    value_paths = {
        "memory_after_sa_construction": [
            "data",
            "root",
            "mem_after_sa_construction",
            "statistics",
            "gather",
        ],
    }
    value_paths.update(key_value_paths)
    df = []
    for exp in experiment_names:
        experiment_path_ = Path(experiment_path) / exp
        if not experiment_path_.exists():
            print("The experiment directory {} doesn't exist".format(experiment_path_))
            sys.exit(1)
        df.append(
            logparser.read_logs_from_directory(
                experiment_path_, "intel", value_paths, glob_pattern="*counter.json"
            )
        )

    df = pd.concat(df)
    # df = df.query("iteration > 0")
    return df


def cast_columns_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype(int)
    return df


def generate_col_plot(df, hue_cols, style_cols, y="total_time", y_min=None):
    hue = df.apply(lambda row: ", ".join(str(row[col]) for col, _ in hue_cols), axis=1)
    hue.name = ", ".join(colname for _, colname in hue_cols)

    style = df.apply(
        lambda row: ", ".join(str(row[col]) for col, _ in style_cols), axis=1
    )
    style.name = ", ".join(colname for _, colname in style_cols)
    g = sns.relplot(
        data=df,
        x="p",
        y=y,
        col="datasize",
        hue=hue,
        markers=True,
        style=style,
        errorbar="pi",
        kind="line",
        dashes=False,
        col_wrap=4,
        facet_kws={"sharey": False, "sharex": True},
    )
    tick_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if y_min != None:
        for ax in g.axes.flat:
            ax.set_ylim(y_min, None)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_values))
            ax.xaxis.set_major_formatter(
                ticker.FixedFormatter([str(v) for v in tick_values])
            )
    return g


def transform_dataframe(df):
    int_columns = ["p", "hierarchy_aware", "num_levels", "not_use_mpi_colls"]
    df = cast_columns_to_int(df, int_columns)
    df["p"] = df["p"].apply(lambda entry: entry / 48)
    df["num_levels"] = df.apply(
        lambda entry: 2 if entry.hierarchy_aware else entry.num_levels, axis=1
    )
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", help="path to base directory where the experiments are located."
    )
    parser.add_argument("--experiments", nargs="*", help="name of the experiments")
    parser.add_argument("--output-name", help="Output file name", default=None)
    parser.add_argument(
        "--mpi-type",
        help="MPI Type for which to filter. If omitted, input will not be filtered for MPI implementation.",
        default=None,
    )
    parser.add_argument(
        "--output_format", choices=["pdf", "pgf", "both"], default="pdf"
    )
    args = parser.parse_args()
    output_path = args.output_name
    if output_path == None:
        output_path = "_".join(args.experiments)
        output_path = output_path + ".pdf"

    print(args.experiments)
    df = parse_logs(args.path, args.experiments)

    transform_dataframe(df)

    pd.set_option("display.max_columns", None)

    pdf = PdfPages(output_path)

    hue_cols = [
        ("not_use_mpi_colls", "RBC-Colls"),
        ("num_levels", "Levels"),
        ("hierarchy_aware", "HAware"),
    ]
    style_cols = [
        ("not_use_mpi_colls", "RBC-Colls"),
        ("hierarchy_aware", "HAware"),
    ]
    # graph_list = [
    #    "gnm-undirected",
    #    "rgg2d",
    #    "rgg2d-permuted",
    #    "rgg3d",
    #    "rgg3d-permuted",
    # ]
    # grouped_df = grouped_df.query("graph in @graph_list")
    # grouped_df = grouped_df.query(
    #    "algorithm != 'ADAPTIVE_ASYNC_SSSP' or active_req_ratio == 0.1"
    # )
    # grouped_df = grouped_df.query(
    #    "algorithm != 'DENSE_DELTA_STEPPING' or (interleaved_exchange == '1' and min_requests_per_pe == '500')"
    # ).reset_index()
    g = generate_col_plot(
        df=df,
        hue_cols=hue_cols,
        style_cols=style_cols,
        y_min=0,
    )

    g.figure.suptitle("all algos")
    g.figure.subplots_adjust(top=0.9)
    pdf.savefig(g.figure, bbox_inches="tight")
    hue_cols = [
        ("not_use_mpi_colls", "RBC-Colls"),
        ("hierarchy_aware", "HAware"),
    ]
    style_cols = [
        ("not_use_mpi_colls", "RBC-Colls"),
        ("hierarchy_aware", "HAware"),
    ]
    g = generate_col_plot(
        df=df.query("num_levels != 1 or hierarchy_aware == 1"),
        hue_cols=hue_cols,
        style_cols=style_cols,
        y_min=0,
    )

    g.figure.suptitle("all algos")
    g.figure.subplots_adjust(top=0.9)
    pdf.savefig(g.figure, bbox_inches="tight")
    pdf.close()


if __name__ == "__main__":
    main()
