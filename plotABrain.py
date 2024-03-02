# Import the necessary files

import sys
import mne
import numpy as np
import matplotlib.pyplot as plt

# Import the evoked file

data_path = mne.datasets.sample.data_path() / "MEG" / "Sample"
bird_fname = data_path / "sample_audvis-ave.fif"
raw = mne.read_evokeds(bird_fname, baseline=None, proj=True, verbose=False)

# Call the conditions and convert it into a dictionary

cods = ("aud/left", "aud/right", "vis/left", "vis/right")
evks = dict(zip(cods, raw))

# Highlight the trans file (goes from mri->meg)

subjects_dir = data_path.parents[1] / "subjects"
trans_file = data_path / "sample_audvis_raw-trans.fif"

# Make the field map. Note that the ch_type can be changed to "meg" as well. 
# It is also not a required parameter, which means that it can be removed without harming the script's functionality.

map = mne.make_field_map(
  evks["aud/left"], ch_type="eeg", trans_file = str(trans_file), subjects_dir=subjects_dir, mode="accurate"
)
evks["aud/left"].plot_field(map, time=0.1)
