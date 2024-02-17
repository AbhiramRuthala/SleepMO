import mne
import sys

source = mne.datasets.sample.data_path()
evoked_file = source / "MEG" / "Sample" / "sample_audvis-ave.fif"
evoked_list= mne.read_evokeds(
    evoked_file, baseline=(None, 0), proj=True, verbose=False
)

conds = ("aud/left","aud/right","vis/left","vis/right")
evks = dict(zip(conds, evoked_list))

graph_type = input("Graph type: ")
datatype = input("Condition: ")
picks_selector = input("Data type: ")

for cod in graph_type:
    if graph_type == "plot":
        evks[datatype].plot(picks=picks_selector, spatial_colors=True, exclude=[], gfp=True)
        sys.exit()
    elif graph_type == "plot topograph":
        evks[datatype].plot_topomap(ch_type=picks_selector, colorbar=True)
        sys.exit()
    elif graph_type == "plot joint":
        evks[datatype].plot_joint(picks=picks_selector)
        sys.exit()
    elif graph_type == "plot diff":
        def customFunc(x):
            return x.max(1)

        for combine in ("mean", customFunc):
            mne.viz.plot_compare_evokeds(evks, picks=picks_selector, combine=combine)

    else:
        sys.exit()
