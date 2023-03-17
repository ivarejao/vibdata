# Code made in Pycharm by Igor Varejao
import logging
import threading
import multiprocessing
import time

import pandas as pd
from typing import List, Tuple
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from vibdata.datahandler.base import RawVibrationDataset
import vibdata.datahandler as datahandler

class ListConcurrency():

    def __init__(self):
        self.lock = threading.Lock()
        self.list = []

    def add(self, element):
        self.lock.acquire()
        try:
            self.list.append(element)
        finally:
            self.lock.release()

def worker(datasets : ListConcurrency, dataset_class, name):
    dt = dataset_class('/tmp/', download=True)  # Instantiate
    # print(f"Working {name}")
    datasets.add(dt)
    # print(f"Datasets: [{datasets.datasets}]")
    print(f"End thread {threading.current_thread().name}")

def load_datasets() -> List[RawVibrationDataset]:
    datasets = {
        "CWRU": datahandler.CWRU_raw,
        # "EAS": datahandler.EAS_raw,
        "IMS": datahandler.IMS_raw,
        # "MAFAULDA": datahandler.MAFAULDA_raw,
        "MFPT": datahandler.MFPT_raw,
        # "PU": datahandler.PU_raw,
        "RPDBCS": datahandler.RPDBCS_raw,
        "UOC": datahandler.UOC_raw,
        "XJTU": datahandler.XJTU_raw,
        "SEU": datahandler.SEU_raw,
    }
    # Instantiate each dataset
    starttime = time.time()
    print("Loading datasets")
    threads = []
    result = ListConcurrency()
    for name, dt in datasets.items():
        t = threading.Thread(target=worker, args=(result, dt, name), name=name)
        threads.append(t)
        t.start()

    # Join threads
    for t in threads:
        t.join()
    print('That took {} seconds'.format(time.time() - starttime))
    return result.list

def worker_plot(dataset : RawVibrationDataset, samples_plot : ListConcurrency,
                size_plot : ListConcurrency, rotation_plot : ListConcurrency,
                sample_rate_plot : ListConcurrency):
    print(f"Creating plots for {dataset.name()}")
    metainfo = dataset.getMetaInfo(labels_as_str=True)
    sizes = np.array([signal['signal'][0].size for signal in dataset])
    metainfo['dataset'] = dataset.name()
    metainfo['size'] = sizes
    metainfo['samples'] = np.floor(sizes / 1024).astype(int)
    samples_plot.add(
        go.Histogram(
            x=metainfo['label'],
            y=metainfo['samples'],
            histfunc="sum",
            name=dataset.name()
        ),
    )

    size_plot.add(
        go.Histogram(
            x=metainfo['size'],
            y=metainfo['samples'],
            histfunc="sum",
            name=dataset.name()
        )
    )

    if 'rotation_hz' in metainfo.columns:
        rotation_plot.add(
            go.Histogram(
                x=metainfo['rotation_hz'],
                y=metainfo['samples'],
                histfunc="sum",
                name=dataset.name()
            )
        )

    if 'sample_rate' in metainfo.columns:
        sample_rate_plot.add(
            go.Histogram(
                x=metainfo['sample_rate'],
                y=metainfo['samples'],
                histfunc="sum",
                name=dataset.name()
            )
        )


    print(f"End thread {threading.current_thread().name}")

def plot_sample_distribution(datasets : List[RawVibrationDataset]) -> Tuple[Figure, Figure, Figure, Figure]:

    samples_plot = ListConcurrency()
    size_plot = ListConcurrency()
    rotation_plot = ListConcurrency()
    sample_rate_plot = ListConcurrency()

    threads = []
    for dt in datasets:
        t = threading.Thread(target=(worker_plot), args=(dt, samples_plot, size_plot, rotation_plot, sample_rate_plot),
                             name=f'Plot {dt.name()}')
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    fig = go.Figure()
    fig_signals_size = go.Figure()
    fig_freq_dist = go.Figure()
    fig_sample_rate = go.Figure()

    for plot in samples_plot.list:
        fig.add_trace(plot)

    for plot in size_plot.list:
        fig_signals_size.add_trace(plot)

    for plot in rotation_plot.list:
        fig_freq_dist.add_trace(plot)

    for plot in sample_rate_plot.list:
        fig_sample_rate.add_trace(plot)

    fig.update_layout(
        # barmode="overlay",
        bargap=0.3
    )
    return fig, fig_signals_size, fig_freq_dist, fig_sample_rate

def app(datasets : List[RawVibrationDataset]):
    signal_dist, size_dist, freq_dist, sr_dist = plot_sample_distribution(datasets)
    st.write('''
        # Vibdata Anlysis
        ### Signal samples distribution
    ''')
    st.plotly_chart(signal_dist)
    st.write('### Signal size distribution')
    st.plotly_chart(size_dist)
    st.write('### Frequency rotation distribution')
    st.plotly_chart(freq_dist)
    st.write('### Sample rate distribution')
    st.plotly_chart(sr_dist)

def main():
    datasets = load_datasets()
    app(datasets)

if __name__ == "__main__":
    main()
