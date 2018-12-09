# amt

CS598PS course project - Automatic Music Transcription

## Data Preparation

1. Download the data from https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz
2. Merge train_data and test_data into a new folder called data. Do the same for labels.
3. Create a binaries directory in musicnet. It will hold computed variables so that we can reuse without recomputing.

Final directory structure looks like this (listing only directories except metadata.csv).

.
├── data
│   └── musicnet
│       ├── data/
│       ├── binaries/
│       ├── metadata.csv
│       └── labels/
└── src
