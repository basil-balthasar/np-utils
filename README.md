# np-utils

This repository hosts a collection of utilities intended to be used by Neuroplatform users.

## Installation

The use of this code requires access to the FinalSpark Neuroplatform. 

Please contact us if you have a request pertaining to using these utilities in your Neuroplatform project.

## Contributions and issues

We welcome any feedback, issues or contributions to this repository. Please use the Issues tab to report any problems or suggest improvements.

## Documentation

TBA

## Contents

### StimParamLoader

This utility is used to help you setup and send your stimulation parameters to our system.

**NOTE for deployment:** Make sure the MEA schema image is placed appropriately in the same folder as the .py, and that the path is correctly set in the code.

Features :

- Automated checking of your parameters for errors or redundancies
- An interactive preview of your parameters, to easily keep track of what you are sending and spot any mistakes
- A simple way to send your parameters to the system
- A simple way to disable all parameters currently on the system, to help you properly finish your experiment

![Parameters preview](images/StimParamLoader/param_loader_demo.png)
*Example of the parameters preview*

### SpikeSorting

This utility is used to perform a simple spike sorting operation on the raw spiking data in our database.

This will assign each spike to a cluster, which can then be used for your data analysis.

It requires a certain number of minimum events for proper operation; you may want to adjust the number of components

Features :

- Automated artifact/spike discrimination
- Spike sorting using :
  - ICA or PCA (dimensionality reduction)
  - HDBSCAN or OPTICS (clustering)
- Several plots of the resulting clusters
  - Plot of average trace for each cluster
  - 3D plot of the clusters in the PCA/ICA latent space
  - Plot of all raw traces for a selection of clusters

![Spike sorting](images/SpikeSorting/spike_sorting_clusters_lineplot.png)<br>
*Example of a line plot of the average trace for each cluster*

![Spike sorting](images/SpikeSorting/spike_sorting_latent_space_plot.png)<br>
*Example of a 3D plot of the clusters in the PCA/ICA latent space*

![Spike sorting](images/SpikeSorting/spike_sorting_raw_lineplot.png)<br>
*Example of a line plot of all raw traces for a selection of clusters*

### RawRecordingLoader

A tool to load raw recordings from the Neuroplatform database.

*Note : if you wish to perform raw recordings, please contact us to get access to the raw recording feature. In the future, this will be accessible to all Neuroplatform users, but currently requires manual adjustments.*

Features :

- Load raw recordings from an h5 file
  - Choose which channels to load
  - For memory-heavy recordings, specify either the index or the time range of the recording you wish to load
