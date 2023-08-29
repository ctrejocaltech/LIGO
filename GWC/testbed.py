import pandas as pd
import altair as alt
import numpy as np
import streamlit as st
import panel as pn
import json
import requests

response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)

print(todos)