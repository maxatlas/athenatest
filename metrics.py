import config as c
from pandas import DataFrame

from pathlib import Path


def get_mce(y_pred, y_true, benchmark_session_id="") -> DataFrame:
    if not y_pred:
        return DataFrame(["Bad session: Deviated more than %f from session %s results" %
                          (c.deviation_threshold, benchmark_session_id)])
    mce = DataFrame([])
    return mce


def get_ece(y_pred, y_true, benchmark_session_id="") -> DataFrame:
    if not y_pred:
        return DataFrame(["Bad session: Deviated more than %f from session %s results" %
                          (c.deviation_threshold, benchmark_session_id)])
    ece =  DataFrame([])
    return ece


def get_confusion_matrix(y_pred, y_true, benchmark_session_id="") -> DataFrame:
    if not y_pred:
        return DataFrame(["Bad session: Deviated more than %f from session %s results" %
                          (c.deviation_threshold, benchmark_session_id)])
    cm = DataFrame([])
    return cm


def plot_mce(df: DataFrame):
    # save fig in folder
    return


def plot_ece(df: DataFrame):
    # save fig in folder
    return


def plot_confusion_matrix(df: DataFrame):
    # save fig in folder
    return

