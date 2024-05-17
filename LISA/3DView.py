# Standard libraries
import datetime
import math
import pathlib
import sqlite3
import sys

# Installed libraries
import spiceypy

# Lisa Tools
import logging

import ligo.skymap.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lisacattools import convert_ecliptic_to_galactic
from lisacattools import HPhist
from lisacattools import OFF
from lisacattools.catalog import GWCatalogs
from lisacattools.catalog import GWCatalogType

# Append to root directory of this repository
sys.path.append("../")

logger = logging.getLogger("lisacattools")
logger.setLevel(
    OFF
)  # Set the logger to OFF. By default, the logger is set to INFO

# Start by loading the main catalog file processed from GBMCMC outputs
catPath = "/Users/tutorial/data/ucb/"
catalogs = GWCatalogs.create(GWCatalogType.UCB, catPath, "UCB.h5")
catalog = catalogs.get_last_catalog()
detections_attr = final_catalog.get_attr_detections()
detections = final_catalog.get_detections(detections_attr)

# loop over all sources in catalog and append chain samples to new dataframe
sources = list(catalog.get_detections())

print(sources)