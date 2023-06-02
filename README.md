# GPQPS - General-Purpose Query Processing on Summary graphs

This is a companion code repository for the paper: 

Aris Anagnostopoulos, Valentina Arrigoni, Francesco Gullo, Giorgia Salvatori, Lorenzo Severini, "[*General-purpose Query Processing on Summary Graphs*](pdf/GPQPS_extended.pdf)"


## License
Everything in this repository is distributed under the Apache License, version 2.0. See the [LICENSE](LICENSE) file and the [NOTICE](NOTICE) file.


## Contacts
For any question, please contact [Francesco Gullo](mailto:gullof@acm.org)

## Requirements
* `Python v3.8+`
* `Jupyter Notebook v6.0+`
* [`NetworkX`](https://networkx.org/)`v3.1
* [`SciPy`](https://scipy.org/)`v1.10`
* [`NumPy`](https://numpy.org/)`v1.24`
* [`scikit-learn`](https://scikit-learn.org/stable/)`v1.2` 
* [`pandas`](https://pandas.pydata.org/)`v2.0`


## Usage

### Generating summary graphs

* [S2L](https://doi.org/10.1007/s10618-016-0468-8) custom implementation (used for directed and/or edge-weighted graphs, as well as smaller graphs): 
	- Available in the [`Dump-summary-S2L.ipynb`](nb/Dump-summary-S2L.ipynb) notebook
	- We follow the original description of [S2L](https://doi.org/10.1007/s10618-016-0468-8), without the sketching and approximate distance computation. We performed *k-Means* on the adjacency matrix of the input graph using the classical Lloyd algorithm and *$k$-Means++* inizialization, with `tol=0.0001` and maximum number of iterations equal to `20`

* [S2L](https://doi.org/10.1007/s10618-016-0468-8) official implementation (used for larger graphs):
   - We use the code available [here](https://github.com/rionda/graphsumm)
   - Parameters used: `./summ -k <#OUTPUT_SUPERNODES> -b -t 2 -m 20 -d 1000 <GRAPH_FILE>
   - After that, `.pickle` summaries are generated with the [`Dump-summary-S2L_large-graphs.ipynb`](nb/Dump-summary-S2L_large-graphs.ipynb) notebook

* [SWeG](https://doi.org/10.1145/3308558.3313402):
   - We use the (unofficial) implementation available [here](https://github.com/MahdiHajiabadi/GSCIS_TBUDS)
   - SWeG summaries are loaded with the `load-sweg-summary` function in [`BatchQueryProcessing.ipynb`](nb/BatchQueryProcessing.ipynb.ipynb)

### Query processing

* Process and evaluate queries with the GPQPS methods described in the [paper](pdf/GPQPS_extended.pdf) (i.e., *Naive-GPQPS* and *Probabilistic-GPQPS*): [`BatchQueryProcessing.ipynb`](nb/BatchQueryProcessing.ipynb.ipynb) notebook

### Other custom implementations of well-known algorithms
* ...
* ...
