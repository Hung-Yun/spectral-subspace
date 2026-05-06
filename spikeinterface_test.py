#%%

import spikeinterface.extractors as se

# Load the NS5 file
infile = '/Users/mobeets/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/Data/RiverRaid/YFU/YFU_0221_riverraid/neural/EMU-0221_riverraid_NSP-2.ns5'
recording = se.read_blackrock(infile)

print(recording)

#%% visualize photodiode and audio

import matplotlib.pyplot as plt
import spikeinterface.widgets as sw
import numpy as np

channel_ids = recording.get_channel_ids()
channel_names = recording.get_property("channel_name")
channel_name_to_id = {name: id_ for name, id_ in zip(channel_names, channel_ids)}

bin_size = 100 # in ms
t1 = recording.get_start_time()
t2 = recording.get_end_time()
t_start = t1 + 95

for name in ['Audio', 'Photodiode']:
    ch_id = channel_name_to_id[name]
    print(ch_id)

    # can get trace this way, but it will have 30k samples per second
    trace = recording.get_traces(channel_ids=[ch_id])
    fs = recording.get_sampling_frequency()
    trace_100ms = trace[::int(fs / bin_size)]

    # or you can just use the built-in function to visualize traces
    sw.plot_traces(recording, channel_ids=[ch_id], time_range=(t_start, t_start+0.5))
plt.axis('tight')

#%% add dummy probe

import numpy as np
from probeinterface import generate_linear_probe, ProbeGroup
probegroup = ProbeGroup()
for i in range(1):
    probe = generate_linear_probe(num_elec=8)
    probe.set_device_channel_indices(np.arange(8) + i * 8)
    probegroup.add_probe(probe)
recording_one_probe = recording.set_probegroup(probegroup, group_mode='by_probe')

#%% preprocess

import spikeinterface.preprocessing as sp

# Bandpass filter between 300–6000 Hz
recording_f = sp.bandpass_filter(recording_one_probe, freq_min=300, freq_max=6000)

# Common median reference
recording_cmr = sp.common_reference(recording_f, reference='global')

#%%

import spikeinterface.sorters as ss
print(ss.available_sorters())

# Run spike sorter
sorting = ss.run_sorter(
    sorter_name='tridesclous',
    recording=recording_cmr,
    # output_folder='data/ks_output',
    verbose=True
)

#%%

print(sorting)
print("Unit IDs:", sorting.get_unit_ids())
print("First few spike times of first unit:", sorting.get_unit_spike_train(sorting.unit_ids[0])[:10])

#%%

from spikeinterface import create_sorting_analyzer
analyzer = create_sorting_analyzer(sorting=sorting, recording=recording_one_probe, format="memory")

#%%

analyzer.compute(
    "random_spikes",
    method="uniform",
    max_spikes_per_unit=500,
)
analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
analyzer.compute("templates", operators=["average", "median", "std"])

#%%

analyzer.compute("templates", operators=["average", "median", "std"])

#%%
import matplotlib.pyplot as plt

ext_templates = analyzer.get_extension("templates")
av_templates = ext_templates.get_data(operator="average")

for unit_index, unit_id in enumerate(analyzer.unit_ids):
    fig, ax = plt.subplots()
    template = av_templates[unit_index]
    ax.plot(template)
    ax.set_title(f"{unit_id}")

#%%


import spikeinterface.widgets as sw

# View spike rasters or waveforms
# sw.plot_rasters(sorting)                     # if you have sorting + event times
sw.plot_unit_waveforms(analyzer)   # visualize waveforms per unit
