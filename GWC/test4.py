import pandas as pd
import altair as alt
import numpy as np
import streamlit as st
import panel as pn
import json
import solara



pn.extension(design='material')

csv_file = ("https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv")
data = pd.read_csv(csv_file, parse_dates=["date"], index_col="date")

data.tail()

