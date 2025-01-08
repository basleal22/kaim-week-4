import pandas as pd
import numpy as np

def promo_dist(data):
    dist=data['promo'].value_counts(normalize=True)
    return dist