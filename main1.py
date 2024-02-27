# Input necessary files

import matplotlib.pyplot as plt
import mne
import numpy as np

from scipy.stats import ttest_rel
import sys

# Input the individual's data

name = input("Name: ")

subject_number = input("Subject Number: ")
Session_number = input("Session Type: ")
SleepSchedule = input("What sleep schedule would you like to follow? ")

# Load the data 
data_path = mne.datasets.ssvep.data_path()
bird_fname = data_path / f"sub-{subject_number}" / f"ses-{Session_number}" / "eeg" / f"sub-{subject_number}_ses-{Session_number}_task-ssvep_eeg.vhdr"

# Load the raw data
data_path2 = mne.datasets.sample.data_path()
new_file = data_path2 / "MEG" / "Sample" / "sample_audvis_raw.fif"
raw2 = mne.io.read_raw_fif(new_file, preload=True)
raw2.crop(60)

# Read .vhdr file and preprocess it.
raw = mne.io.read_raw_brainvision(bird_fname, verbose=False, preload=True)
raw.info["line_freq"] = 50.0

original_raw = raw2.copy()
new_raw = raw2.apply_hilbert()
#new_raw2 = mne.preprocessing.ICA(raw2)
#new_raw3 = raw.maxwell_filter()
new_raw2 = mne.preprocessing.maxwell_filter(raw2)

age = input("Age: ")

# Compare data types between hilbert filter, maxwell filter, and the original raw file
print(f"This is the original file data type: {original_raw.get_data().dtype}. This is the new file data type: {new_raw.get_data().dtype}. The file type after conducting a Maxwell Filter is {new_raw2.get_data().dtype}.")

montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

# Reference the EEG data
raw.set_eeg_reference("average", projection=False, verbose=False)

# Filter the data to improve signal quality
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

# Define potential frequencies that can be used for identification.
event_id = {"12hz": 255, "15hz": 155}
events, _ = mne.events_from_annotations(raw, verbose=False)
tmin, tmax = -0.1, 20.0
baseline=None
epochs = mne.Epochs(raw, events=events, baseline=baseline, verbose=False, tmax=tmax, tmin=tmin, event_id=[event_id["12hz"], event_id["15hz"]],)

tmin = 1.0
tmax = 20.0
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]

# Compute PSD (Power Spectral Density)
# PSD is the spread of power produced by EEG series within the frequency domain. 

spectrum = epochs.compute_psd(
  # Use the Welch method to compute PSD.
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)

# Define a kernel to calculate the average power of the neighboring frequency bins.
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):

  # Define certain variables that correspond to certain frequency bin types.
  
    average_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),

        )
    )
    average_kernel /= average_kernel.sum()
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, average_kernel, mode="valid"), axis=-1, arr=psd

    )

    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)


# Graph

# Plot PSD
fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0,1))[freq_range]
psds_std = psds_plot.std(axis=(0,1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="blue")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, alpha=0.2, color="blue"
)
axes[0].set(title="PSD Spectrum", ylabel="Power Spectral Density [dB]")

# Plot SNR. SNR is the Signal-To-Noise Ratio of EEG data. The signal is the power produced by a given frequency bin, and the noise is the average power of the neighboring frequency bins.
# Preprocessing the data helps with identifying proper SNR values through higher signal quality.

snr_mean = psds_plot.mean(axis=(0,1))[freq_range]
snr_std = psds_plot.std(axis=(0,1))[freq_range]
axes[1].plot(freqs[freq_range], snr_mean, color="red")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, alpha=0.2, color="red"
)
axes[1].set(title="SNR Spectrum", xlabel="Frequency [dB]", ylabel="Signal-To-Noise Ratio", ylim=[-2, 30], xlim=[fmin, fmax])
fig.show()

# Plot both graphs to identify differences between PSD and SNR.

# Identify stimulation frequency value.
stim_freq = 12.0

# Use stim_freq values to identify frequency bins.
i_bin_12hz = np.argmin(abs(freqs - stim_freq))

i_bin_24hz = np.argmin(abs(freqs - 24))
i_bin_36hz = np.argmin(abs(freqs - 36))
i_bin_15hz = np.argmin(abs(freqs - 15))
i_bin_30hz = np.argmin(abs(freqs - 30))
i_bin_45hz = np.argmin(abs(freqs - 45))

i_identify_trial_12hz = np.where(epochs.events[:, 2] == event_id["12hz"])[0]
i_identify_trial_15hz = np.where(epochs.events[:, 2] == event_id["15hz"])[0]

# Identify potential visual ROI
roi_vis = [
    "POz",
    "Oz",
    "O1",
    "O2",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "PO9",
    "PO10",
    "O9",
    "O10",
]  # visual roi

picks_from_roi_viz = mne.pick_types(
    epochs.info, exclude="bads", eeg=True, stim=False, selection=roi_vis,
)

snrs_roi = snrs[i_identify_trial_12hz, :, i_bin_12hz][:, picks_from_roi_viz]
average_snr = int(snrs_roi.mean())
print(f"Subject 2: SNR Data from 12hz trial")
value_szn = raw.info["bads"]
print(f"Average SNR (ROI): {snrs_roi.mean()}")
print(f"Rounded version: {average_snr}")

SNR_predict_melatonin ={
    41:"75",
    25:"43",
    11:"25",
    15:"30"

}

# Acquire the melatonin levels

melatonin_get = SNR_predict_melatonin.get(average_snr, " ")

report = mne.Report(title=name+"'s Data Analysis")
new_raw.pick(picks=["eeg"]).crop(tmax=100).load_data()
report.add_raw(raw=new_raw, title="Report")
report.save("report_raw.html", overwrite=True)
print(f"We are generating {name}'s data report")

# Define the logistics of the recommendations

if melatonin_get == "75":
    myhtml=f"""
    <h1> {name}'s Personalized Sleep Recommendations </h1>
    <p>
    Through computed <b>melatonin</b> levels, here's what we suggest you do:
    <p></p>
    <p>
    1. Make sure to stay hydrated
    </p>
    <p></p>
    <p>
    2. Make sure to limit the exposure to blue light before you sleep
    </p>
    <p></p>
    <p>
    3. Make sure to drop your body temperature before you sleep
    </p>
    """
  # Save the recommendation to the initial file.
    report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
    report.save("report_add_html.html", overwrite=True)
elif melatonin_get == "25":
    myhtml = f"""
        <h1> {name}'s Personalized Sleep Recommendations </h1>
        <p>
        Through computed <b>melatonin</b> levels, here's what we suggest you do:
        <p></p>
        <p>
        1. Make sure to stay hydrated
        </p>
        <p></p>
        <p>
        2. Make sure to limit the exposure to blue light before you sleep
        </p>
        <p></p>
        <p>
        3. Make sure to drop your body temperature before you sleep
        </p>
        """
    report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
    report.save("report_add_html.html", overwrite=True)
else:
  # System will exit if corresponding SNR isn't reached. I need to define what this means for an individual.
    sys.exit()

report.save("report_add_html.html", overwrite=True)
print(f"{name}'s data report has been generated!")

# Define other logistics for the recommendations

def sleepHTMLGenerator():
  # Check for the hours inputted.
    if SleepSchedule == "10 hours":
      # Check for certain data parameters.
        if age < 18:
            myhtml = f"""
            <h1> Great Job {name}! </h1>
            <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml="""
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        myhtml="""
        <h1> 9 hours is a good schedule to accomodate to! </h1>
        <p> We will help you every step of the way! </p>
        """
        report.add_html(title=name+"'s Personalized Sleep Recommendations", html=myhtml)
        report.save("report_add_html.html", overwrite=True)
        if age == 18:
            myhtml = """
                <h1> Good Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 9 hours is a great amount of sleep </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age <"45":
            myhtml="""
            <h1> Wow </h1>
            <p> That's a good amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml = """
            <h1> 7 hours? Let's see if you can get more </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> 7 hours? Good job, {name}! </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> 7 hours? Let's see if you can get more </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = f"""
            <h1> Here are your sleep recommendations, {name} </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "6 hours":
        myhtml="""
        <h1> You should aim for more sleep </h1>
        <p> 6 hours isn't the most sustainable amount of sleep </p>"""
        report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if age == 18:
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age <"45":
            myhtml="""
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "12 hours":
        myhtml="""
        <h1> Might need to select a higher schedule </h1>
        """
        report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if age == 18:
            myhtml="""
            <h1> Try to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml="""
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        myhtml = """
                    <h1> Might need to select a higher schedule </h1>
                    """
        report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if age == 18:
            myhtml = """
                        <h1> Try to aim for more sleep </h1>
                        """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    else:
        pass

# Call the function.

sleepHTMLGenerator()
