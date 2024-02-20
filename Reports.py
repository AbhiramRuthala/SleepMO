import matplotlib.pyplot as plt
import mne
from pathlib import Path
import sys
import numpy as np
import scikit.ndimage
import templib

name = input("Name: ")

source = Path(mne.sample.datasets.data_path(verbose=False))
sample_dir = source / "MEG" / "Sample"
subjects_dir = source / "Subjects"

raw_path = sample_dir / "sample_audvis_raw.fif"
raw = mne.io.read_raw(raw_path)
raw.pick(picks=["eeg","mag","grad"]).crop(tmax=100).load_data()

report = mne.Report(title=name+"'s"+" Data Report")
report.add_raw(raw=raw, psd=False, title="Raw Data Analysis")
report.save("report_raw.html", overwrite=True)
