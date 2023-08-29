import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
#import django
#django.setup()
from event.models import Event
from tile.models import TileDataset
from parameter.models import Parameter
#from catalog.api_classbased import get_strain_files
from gwpy.timeseries import TimeSeries



qs = Event.objects.all()

print(qs)
