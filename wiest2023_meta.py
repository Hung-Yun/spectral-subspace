import pandas as pd

# The released Med_OFFs.m and Med_ONs.m files contain three separate positional
# objects: an alphabetically ordered file list, a MATLAB 1-indexed channel
# vector, and a start/stop-time matrix.  There is no participant metadata table
# joining them.  PAIR_SPECS makes that implicit positional join explicit:
#
#   1. Match row N of each channel/time array to file N in the author's order.
#   2. Subtract one from each MATLAB channel number to store a Python 0-indexed
#      row of SmrData.WvData as channel_ix_off/channel_ix_on.
#   3. Read that row's SmrData.WvTits entry and store its exact uploaded label
#      as physical_loc_off/physical_loc_on (for example, L02 or R13).
#   4. Keep the authors' time boundaries unchanged.  Their MATLAB slice
#      start*fs:stop*fs includes both endpoints; load_data preserves
#      that behavior.
#
# This is deliberately hard-coded provenance, not a new data-driven channel
# selection.  PAIR_SPECS retains all 30 published selections.  pair_specs below
# is the analysis DataFrame: it keeps only matching, explicitly identified
# physical labels.  Generic "Virtual" labels are excluded because equality of
# that string cannot prove that the underlying contact pair is the same.
PAIR_SPEC_ROWS = [
        ("G10_leSTN", "G10_leSTN_OFF.mat", 8, "Le02", 760, 820, "G10_leSTN_ON.mat", 8, "Le02", 1, 61),
        ("G10_riSTN", "G10_riSTN_OFF.mat", 10, "R02", 2360, 2420, "G10_riSTN_ON.mat", 9, "R02", 1, 61),
        ("G23_leSTN", "G23_leSTN_OFF.mat", 9, "L13", 1, 61, "G23_leSTN_ON.mat", 9, "L13", 70, 130),
        ("G23_riSTN", "G23_riSTN_OFF.mat", 10, "R02", 1550, 1610, "G23_riSTN_ON.mat", 10, "R02", 1460, 1520),
        ("G24_leSTN", "G24_leSTN_OFF.mat", 8, "L02", 1, 61, "G24_leSTN_ON.mat", 8, "L02", 1, 61),
        ("G24_riSTN", "G24_riSTN_OFF.mat", 10, "R02", 1, 61, "G24_riSTN_ON.mat", 10, "R02", 1, 61),
        ("G25_leSTN", "G25_leSTN_OFF.mat", 0, "L24", 1, 61, "G25_leSTN_ON.mat", 0, "L24", 1, 61),
        ("G25_riSTN", "G25_riSTN_OFF.mat", 0, "R24", 1, 61, "G25_riSTN_ON.mat", 0, "R24", 1, 61),
        ("G27_leSTN", "G27_leSTN_OFF.mat", 0, "L13", 1, 61, "G27_leSTN_ON.mat", 29, "Lv13", 277, 337),
        ("G27_riSTN", "G27_riSTN_OFF.mat", 0, "R24", 1, 61, "G27_riSTN_ON.mat", 29, "Rv13", 1, 61),
        ("G28_leSTN", "G28_leSTN_OFF.mat", 6, "L02", 1, 61, "G28_leSTN_ON.mat", 13, "L02", 1, 61),
        ("G28_riSTN", "G28_riSTN_OFF.mat", 9, "R13", 1, 61, "G28_riSTN_ON.mat", 13, "R24", 230, 290),
        ("G30_leSTN", "G30_leSTN_OFF.mat", 8, "L02", 1, 61, "G30_leSTN_ON.mat", 17, "L24", 1, 61),
        ("G30_riSTN", "G30_riSTN_OFF.mat", 6, "R02", 1, 61, "G30_riSTN_ON.mat", 17, "R13", 1, 61),
        ("G31_leSTN", "G31_leSTN_OFF.mat", 6, "L02", 1, 61, "G31_leSTN_ON.mat", 14, "L13", 1, 59),
        ("G31_riSTN", "G31_riSTN_OFF.mat", 9, "R13", 1, 61, "G31_riSTN_ON.mat", 14, "R13", 1, 61),
        ("G32_leSTN", "G32_leSTN_OFF.mat", 5, "L13", 1, 61, "G32_leSTN_ON.mat", 22, "L13", 1, 61),
        ("G32_riSTN", "G32_riSTN_OFF.mat", 6, "R02", 1, 61, "G32_riSTN_ON.mat", 22, "R13", 1, 61),
        ("G33_leSTN", "G33_leSTN_OFF.mat", 19, "Virtual", 1, 61, "G33_leSTN_ON.mat", 24, "Virtual", 1, 61),
        ("G33_riSTN", "G33_riSTN_OFF.mat", 21, "Virtual", 1, 61, "G33_riSTN_ON.mat", 26, "Virtual", 1, 61),
        ("G34_leSTN", "G34_leSTN_OFF.mat", 23, "Virtual", 200, 260, "G34_leSTN_ON.mat", 23, "Virtual", 200, 260),
        ("G34_riSTN", "G34_riSTN_OFF.mat", 25, "Virtual", 200, 260, "G34_riSTN_ON.mat", 25, "Virtual", 200, 260),
        ("K11_riSTN", "K11_riSTN_OFF.mat", 2, "R02", 1, 61, "K11_riSTN_ON.mat", 7, "R13", 1, 61),
        ("K6_leSTN", "K6_leSTN_OFF.mat", 15, "Virtual", 1, 61, "K6_leSTN_ON.mat", 15, "Virtual", 1, 61),
        ("K6_riSTN", "K6_riSTN_OFF.mat", 18, "Virtual", 1, 61, "K6_riSTN_ON.mat", 18, "Virtual", 1, 61),
        ("K7_leSTN", "K7_leSTN_OFF.mat", 11, "L13", 1, 61, "K7_leSTN_ON.mat", 7, "L13", 1, 61),
        ("K8_leSTN", "K8_leSTN_OFF.mat", 6, "L02", 70, 130, "K8_leSTN_ON.mat", 13, "L02", 1, 61),
        ("K8_riSTN", "K8_riSTN_OFF.mat", 8, "R02", 70, 130, "K8_riSTN_ON.mat", 13, "R02", 1, 61),
        ("XG37_leSTN", "XG37_leSTN_OFF.mat", 8, "L24", 1, 61, "XG37_ERNA_leSTN_ON.mat", 19, "L24", 1, 61),
        ("XG39_riSTN", "XG39_riSTN_OFF.mat", 7, "R13", 1, 61, "XG39_ERNA_riSTN_ON.mat", 17, "R13", 1, 61),
]
PAIR_SPECS = pd.DataFrame(PAIR_SPEC_ROWS, columns=("subject_hemi", "file_off", "channel_ix_off", "physical_loc_off", "start_s_off", "stop_s_off", "file_on", "channel_ix_on", "physical_loc_on", "start_s_on", "stop_s_on"))
PAIR_SPECS.index.name = "author_pair_ix"
PAIR_SPECS = PAIR_SPECS.reset_index()
