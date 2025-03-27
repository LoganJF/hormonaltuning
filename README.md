# hormonaltuning
A collection of functions, and examples associated with (Fickling et al., 2025), project ought not be used until ~ April 2025 when documentation, testing, and proper packaging is complete. But, feel free to take a gander now iffin you wish.

How do I use hormonaltuning?
-------------
hormonaltuning is used primarily for spike feature extraction (requiring neural spike which are already detected, see github.com/LoganJF/stns), as well as clustering of these spikes using various approaches. If one is interested in replicating/reproducing/extending the analyses seen in (Fickling et al., 2025), this is a good palce to start!

```python
from hormontaltuning import feature_extraction_preprocessing_pipeline
from time import time

# Set various inputs
filler_val = -100
segment_dur = 200
max_CP_thres = 40
max_time_from_nearest_phase_thres = 5
n_sec_cp_combine = 5
thres_long_burst_dur = 1
neurons_valid = ['IC','PD','AM', 'LG', 'DG']
num_phase_bins = 100


# Set list of experiment datas
SSUS_exps = ['3-20-23', 
             '3-26-23', 
             '4-3-23',
             '4-4-23', 
             '4-17-23']
all_data = []

SSUS_
start_time = time()

for date in SSUS_exps:
    print('Starting exp: ', date)
    
    segs=feature_extraction_preprocessing_pipeline(date, condition = 'Saline 0 +\n gsif 10^-6M', 
                                                   filler_value = -filler_val,
                                                   segment_duration_seconds = segment_dur,
                                                   thres_max_CP_long_IC = max_CP_thres,
                                                   thres_max_time_from_nearest_phase = max_time_from_nearest_phase_thres,
                                                   n_sec_cp_combine = n_sec_cp_combine,
                                                   long_IC_burst_dur = thres_long_burst_dur,
                                                   neurons_keep = neurons_valid,
                                                   phase_bins=num_phase_bins, verbose = True,)
    all_data.append(segs)
print('Total time to process data: ', time()-start_time)
```

See the documentation, or example folder, for more examples, or in python call help(functionname) to pull up the docstrings of the relevant functions

hormonaltuning is currently in development; some features have not yet been implemented.
See 'Development status' below.

Requirements
------------
scipy

numpy

pandas

matplotlib

seaborn

sklearn

stns [will ultimately be removed]


Quick start
-----------

hormonaltuning will be able to be installed using pip [~March 2025]:

    $ pip install hormonaltuning

If you want to run the latest version of the code, you can install from git:

    $mkdir hormonaltuning
    $cd hormonaltuning
    $git clone https://github.com/LoganJF/hormonaltuning.git .
    $python setup.py install


Development status
------------------
Beta testing currently 0.0.1
