import mne
import matplotlib.pyplot as plt
import numpy as np

data_path = mne.datasets.sample.data_path()
sample_data_section = data_path / "MEG" / "Sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_section, preload=True)
raw.crop(tmax=10)

original_raw = raw.copy()
new_raw = raw.apply_hilbert()
print(f"first data type is {original_raw.get_data().dtype} and new data type is {new_raw.get_data().dtype}")

#MelatoninInput = input("Melatonin levels: ")

melatoninRates ={
    "groggy": 25,
    "fit": 50,
}

if new_raw.get_data().dtype == "complex128":
    print("Melatonin levels have been processed")
else:
    pass

#def SleepMOCaptcha():
 #   melatonin1 = melatoninRates.get(MelatoninInput, "Invalid input")

def melatoninPlot():
    plt.figure(figsize=(8,8))
    plt.title("Sleep analysis over time")
    plt.xlabel("Time (In Seconds)")
    plt.ylabel("Melatonin")
    x = np.linspace(0,10,2)
    y = [0, 10]
    plt.plot(x, y, color="pink")
    plt.show()

melatoninPlot()
