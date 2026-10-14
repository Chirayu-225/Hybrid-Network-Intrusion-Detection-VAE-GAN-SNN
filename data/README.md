# Data

Raw data is not committed. Put the CICIDS2017 CSV files (MachineLearningCVE or TrafficLabelling bundle) in:

    data/raw/cicids2017/*.csv

Download from the official source: https://www.unb.ca/cic/datasets/ids-2017.html (cite Sharafaldin, Lashkari and
Ghorbani, ICISSP 2018). For a quick pipeline check without the dataset run `make smoke`, which writes a synthetic
CICIDS-schema stand-in to `data/synthetic/` (never report metrics from it).
