# Code made in Pycharm by Igor Varejao
import pandas as pd
from typing import List, Tuple
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from vibdata.datahandler.base import RawVibrationDataset
import vibdata.datahandler as datahandler

def load_datasets() -> List[RawVibrationDataset]:
    datasets = {
        "CWRU": datahandler.CWRU_raw,
        # "EAS": datahandler.EAS_raw,
        # "IMS": datahandler.IMS_raw,
        # "MAFAULDA": datahandler.MAFAULDA_raw,
        "MFPT": datahandler.MFPT_raw,
        # "PU": datahandler.PU_raw,
        "RPDBCS": datahandler.RPDBCS_raw,
        # "UOC": datahandler.UOC_raw,
        # "XJTU": datahandler.XJTU_raw,
        # "SEU": datahandler.SEU_raw,
    }
    # Instantiate each dataset
    return [dt('/tmp/', download=True) for dt in datasets.values()]

def plot_sample_distribution(datasets : List[RawVibrationDataset]) -> Tuple[Figure, Figure, Figure]:
    fig = go.Figure()
    fig_signals_size = go.Figure()
    fig_freq_dist = go.Figure()
    for dt in datasets:
        metainfo = dt.getMetaInfo(labels_as_str=True)
        sizes = np.array([signal['signal'][0].size for signal in dt])
        metainfo['dataset'] = dt.name()
        metainfo['size'] = sizes
        metainfo['samples'] = np.floor(sizes / 1024).astype(int)
        fig.add_trace(
            go.Histogram(
                x=metainfo['label'],
                y=metainfo['samples'],
                histfunc="sum",
                name=dt.name()
            ),
        )

        fig_signals_size.add_trace(
            go.Histogram(
                x=metainfo['size'],
                y=metainfo['samples'],
                histfunc="sum",
                name=dt.name()
            )
        )

        if 'rotation_hz' in metainfo.columns:
            fig_freq_dist.add_trace(
                go.Histogram(
                    x=metainfo['rotation_hz'],
                    y=metainfo['samples'],
                    histfunc="sum",
                    name=dt.name()
                )
            )

    fig.update_layout(
        # barmode="overlay",
        bargap=0.3
    )
    return fig, fig_signals_size, fig_freq_dist

def app(datasets : List[RawVibrationDataset]):
    st.write('''
        # Vibdata Anlysis
        ### Signal samples distribution
    ''')
    signal_dist, size_dist, freq_dist = plot_sample_distribution(datasets)
    st.plotly_chart(signal_dist)
    st.write('### Signal size distribution')
    st.plotly_chart(size_dist)
    st.write('### Frequency rotation distribution')
    st.plotly_chart(freq_dist)

def main():
    datasets = load_datasets()
    app(datasets)

if __name__ == "__main__":
    main()
