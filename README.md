# GPQPS - General-Purpose Query Processing on Summary graphs

This is a companion code repository for the paper: 

Aris Anagnostopoulos, Valentina Arrigoni, Francesco Gullo, Giorgia Salvatori, Lorenzo Severini, "[*General-purpose Query Processing on Summary Graphs*](pdf/GPQPS_extended.pdf)"


## License
Everything in this repository is distributed under the Apache License, version 2.0. See the [LICENSE](LICENSE) file and the [NOTICE](NOTICE) file.


## Contacts
For any question, please contact [Francesco Gullo](mailto:gullof@acm.org)


## Data
* The smaller datasets and corresponding summaries used in the [paper](pdf/GPQPS_extended.pdf) are available in the [`data`](data) (datasets and [S2L](https://doi.org/10.1007/s10618-016-0468-8) summaries) and [`sweg`](sweg) ([SWeG](https://doi.org/10.1145/3308558.3313402) summaries) folders
* The larger datasets are publicly available: please see the [paper](pdf/GPQPS_extended.pdf) for details on this
* Summaries for the larger datasets can be generated as explained below


## Usage

### Requirements
* `Python v3.9`
* `Jupyter Notebook v6.5.4`
* [`NetworkX`](https://networkx.org/)`v3.1`
* [`SciPy`](https://scipy.org/)`v1.10.1`
* [`NumPy`](https://numpy.org/)`v1.20.3`
* [`scikit-learn`](https://scikit-learn.org/stable/)`v0.24.2` 
* [`pandas`](https://pandas.pydata.org/)`v1.2.4`
* [`tqdm`](https://pypi.org/project/tqdm/)`v4.65.0`

### Generating summary graphs

* [S2L](https://doi.org/10.1007/s10618-016-0468-8) custom implementation (used for directed and/or edge-weighted graphs, as well as smaller graphs): 
	- Available in the [`Dump-summary-S2L.ipynb`](Dump-summary-S2L.ipynb) notebook
	- We follow the original description of [S2L](https://doi.org/10.1007/s10618-016-0468-8), without the sketching and approximate distance computation. We performed `k-Means` on the adjacency matrix of the input graph using the classical Lloyd algorithm and `$k$-Means++` inizialization, with `tol=0.0001` and maximum number of iterations equal to `20`

* [S2L](https://doi.org/10.1007/s10618-016-0468-8) official implementation (used for larger graphs):
   - We use the code available [here](https://github.com/rionda/graphsumm)
   - Parameters used: `./summ -k <#OUTPUT_SUPERNODES> -b -t 2 -m 20 -d 1000 <GRAPH_FILE>`
   - After that, `.pickle` summaries are generated with the [`Dump-summary-S2L_large-graphs.ipynb`](Dump-summary-S2L_large-graphs.ipynb) notebook

* [SWeG](https://doi.org/10.1145/3308558.3313402):
   - We use the (unofficial) implementation available [here](https://github.com/MahdiHajiabadi/GSCIS_TBUDS)
   - SWeG summaries are loaded with the `load_sweg_summary` function in [`Bulk-query-processing.ipynb`](Bulk-query-processing.ipynb)
   - Parameters used to generate SWeG summaries of the experiments in the [paper](pdf/GPQPS_extended.pdf) (`T`: number of iterations; `eps`: error bound):
      + `Facebook` dataset, #supernodes: 708, #superedges: 664 -> `T=300`, `eps=0.54`
      + `Facebook` dataset, #supernodes: 977, #superedges: 2,157 -> `T=100`, `eps=0.72`
      + `Facebook` dataset, #supernodes: 1168, #superedges: 4,448 -> `T=80`, `eps=0.18`
      + `LastFM` dataset, #supernodes: 3,375, #superedges: 1,217 -> `T=300`, `eps=0.72`
      + `LastFM` dataset, #supernodes: 3,568, #superedges: 1,550 -> `T=200`, `eps=0.36`     		
      + `LastFM` dataset, #supernodes: 3,821, #superedges: 1,997 -> `T=80`, `eps=0.54` 
      + `Enron` dataset, #supernodes: 22,832, #superedges: 10,160 -> `T=300`, `eps=0.9`      	
      + `Enron` dataset, #supernodes: 23,957, #superedges: 28,825 -> `T=100`, `eps=0.9`
      + `Enron` dataset, #supernodes: 24,371, #superedges: 44,086 -> `T=80`, `eps=0.18`
      + `AS-Skitter` dataset, #supernodes: 1,087,434, #superedges: 1,704,954 -> `T=100`, `eps=0.72`       
      + `AS-Skitter` dataset, #supernodes: 1,087,430, #superedges: 1,857,562 -> `T=100`, `eps=0.18`
      + `AS-Skitter` dataset, #supernodes: 1,103,931, #superedges: 2,073,416 -> `T=80`, `eps=0.18`

### Query processing

* Process and evaluate queries with the GPQPS methods described in the [paper](pdf/GPQPS_extended.pdf) (i.e., `Naive-GPQPS` and `Probabilistic-GPQPS`): [`Bulk-query-processing.ipynb`](Bulk-query-processing.ipynb) notebook
* Clustering coefficient, PageRank, closeness, community detection, and modularity are computed by the corresponding functions in `NetworkX`; all parameters of those functions are set to their default values; for directed/weighted graphs, the corresponding `NetworkX` functions operating on directed/weighted graphs are used

### Other custom implementations of well-known algorithms
* [Batagelj and Zaversnik’s core decomposition algorithm](https://doi.org/10.1007/s11634-010-0079-y): [`mykcore.py`](src/mykcore.py)
* [Topchy et al.'s clustering aggregation algorithm](https://doi.org/10.1109/ICDM.2003.1250937): `partition_aggregation` function in [`summary_utils.py`](src/summary_utils.py)
