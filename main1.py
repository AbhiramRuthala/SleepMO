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
userInput = input("Brain Mapping Type: ")

data_path = mne.datasets.ssvep.data_path()
bird_fname = data_path / f"sub-{subject_number}" / f"ses-{Session_number}" / "eeg" / f"sub-{subject_number}_ses-{Session_number}_task-ssvep_eeg.vhdr"

data_path2 = mne.datasets.sample.data_path()
new_file = data_path2 / "MEG" / "Sample" / "sample_audvis_raw.fif"
raw2 = mne.io.read_raw_fif(new_file, preload=True)
raw2.crop(60)

raw = mne.io.read_raw_brainvision(bird_fname, verbose=False, preload=True)
raw.info["line_freq"] = 50.0

original_raw = raw2.copy()
new_raw = raw2.apply_hilbert()
#new_raw2 = mne.preprocessing.ICA(raw2)
#new_raw3 = raw.maxwell_filter()
new_raw2 = mne.preprocessing.maxwell_filter(raw2)

age = int(input("Age: "))
specificity = input("""
How specific do you want your recommendations to be? 
1 - Specific
2 - General

Specify your response here by inputting the valid number: """)

print(f"This is the original file data type: {original_raw.get_data().dtype}. This is the new file data type: {new_raw.get_data().dtype}. The file type after conducting a Maxwell Filter is {new_raw2.get_data().dtype}.")

montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

raw.set_eeg_reference("average", projection=False, verbose=False)

raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

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

spectrum = epochs.compute_psd(
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

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):

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

snr_mean = psds_plot.mean(axis=(0,1))[freq_range]
snr_std = psds_plot.std(axis=(0,1))[freq_range]
axes[1].plot(freqs[freq_range], snr_mean, color="red")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, alpha=0.2, color="red"
)
axes[1].set(title="SNR Spectrum", xlabel="Frequency [dB]", ylabel="Signal-To-Noise Ratio", ylim=[-2, 30], xlim=[fmin, fmax])
fig.show()

stim_freq = 12.0

i_bin_12hz = np.argmin(abs(freqs - stim_freq))

i_bin_24hz = np.argmin(abs(freqs - 24))
i_bin_36hz = np.argmin(abs(freqs - 36))
i_bin_15hz = np.argmin(abs(freqs - 15))
i_bin_30hz = np.argmin(abs(freqs - 30))
i_bin_45hz = np.argmin(abs(freqs - 45))

i_identify_trial_12hz = np.where(epochs.events[:, 2] == event_id["12hz"])[0]
i_identify_trial_15hz = np.where(epochs.events[:, 2] == event_id["15hz"])[0]

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

melatonin_get = SNR_predict_melatonin.get(average_snr, " ")

report = mne.Report(title=name+"'s Data Analysis")
new_raw.pick(picks=["eeg"]).crop(tmax=100).load_data()
report.add_raw(raw=new_raw, title="Report")
report.save("report_raw.html", overwrite=True)
print(f"We are generating {name}'s data report")

#def SleepRecommendations():

root = mne.datasets.sample.data_path() / "MEG" / "Sample"
evoked_file = root / "sample_audvis-ave.fif"
evoked_list = mne.read_evokeds(evoked_file, baseline=None, proj=True, verbose=False)

subjects_dir = root.parents[1] / "subjects"
trans_file = root / "sample_audvis_raw-trans.fif"

cods = ("aud/left", "aud/right", "vis/left", "vis/right")
evks = dict(zip(cods, evoked_list))

# %%
# By default, MEG sensors will be used to estimate the field on the helmet
# surface, while EEG sensors will be used to estimate the field on the scalp.
# Once the maps are computed, you can plot them with `evoked.plot_field()
# <mne.Evoked.plot_field>`:

if userInput == "Field map":
    maps = mne.make_field_map(
        evks["aud/left"], trans=str(trans_file), subject="sample", subjects_dir=subjects_dir, ch_type="eeg"
    )
    evks["aud/left"].plot_field(maps, time=0.1)
else:
    pass

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
    report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
    report.save("report_add_html.html", overwrite=True)
    if SleepSchedule == "10 hours":
        if age < 18:
            myhtml = f"""
            <h1> Great Job {name}! </h1>
            <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            else:
                pass

        elif age == 18:
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
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
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml="""
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        if age == 18:
            myhtml = """
                <h1> Good Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> Keep Doing What You're Doing </h1>
            <p> 9 hours is a great amount of sleep </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
            <h1> Wow </h1>
            <p> That's a good amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "6 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "11 hours":
        if age < 18:
            myhtml=f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml=f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml=f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
elif melatonin_get == "45":
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
    if SleepSchedule == "10 hours":
        if age < 18:
            myhtml = f"""
                <h1> Great Job {name}! </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = """
                    <h1> Great Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 10 hours is probably a little too much, but it helps with growth. </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = """
                <h1> Wow </h1>
                <p> That's a great amount of sleep. Great job!
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        if age == 18:
            myhtml = """
                    <h1> Good Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 9 hours is a great amount of sleep </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
                <h1> Wow </h1>
                <p> That's a good amount of sleep. Great job!
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        if age < 18:
            myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "6 hours":
        if age < 18:
            myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        if age < 18:
            myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "11 hours":
        if age < 18:
            myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
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
    if SleepSchedule == "10 hours":
        if age < 18:
            myhtml = f"""
            <h1> Great Job {name}! </h1>
            <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = """
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = """
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        if age == 18:
            myhtml = """
                <h1> Good Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> Keep Doing What You're Doing </h1>
            <p> 9 hours is a great amount of sleep </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
            <h1> Wow </h1>
            <p> That's a good amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com</b>. </i> Find this code on <a href="www.github.com/AbhiramRuthala">  GitHub! </a> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "6 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "11 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General ways to improve sleep")
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
    if SleepSchedule == "10 hours":
        if age < 18:
            myhtml = f"""
            <h1> Great Job {name}! </h1>
            <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

        elif age == 18:
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

        elif age > 18 and age < 27:
            myhtml = """
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = """
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        if age == 18:
            myhtml = """
                <h1> Good Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> Keep Doing What You're Doing </h1>
            <p> 9 hours is a great amount of sleep </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
            <h1> Wow </h1>
            <p> That's a good amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)

            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "6 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "11 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 55 and age < 70:
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
else:
    sys.exit()

report.save("report_add_html.html", overwrite=True)
print(f"{name}'s data report has been generated!")

#SleepRecommendations()

def sleepHTMLGenerator():
    if SleepSchedule == "10 hours":
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

sleepHTMLGenerator()
