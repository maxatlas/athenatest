## Table of Content

- [Project Overview](#project-overview)
  * [Built with](#built-with)
  * [IO Diagram](#io-diagram)
  * [Script Component Diagram](#script-component-diagram)
- [Techniques Employed](#techniques-employed)
  * [Early Stopping](#early-stopping)
  * [Testing Logic](#testing-logic)
  * [False Positives Improvement](#false-positives-improvement)
- [Limitation and Integration](#limitation-and-improvement)
  * [Padding](#padding)
  * [Better Early Stopping](#better-early-stopping)
  * [More metrics](#more-metrics)
  * [Database Integration](#database-integration)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
- [Usage](#usage)
  * [Running](#running)
  * [Testing](#testing)



<!-- ABOUT THE PROJECT -->
## Project Overview
This is a python script to evaluate the performance of a classification model. The script is designed to help a fellow Data Scientist asses the performance of a model and potentially debug / or assure their model’s performance. 

### Built with
[![Python3.10.0](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-3100/)
version 3.10.0

### IO Diagram
![scriptIO](img/scriptIO.png)
The script *main.py* takes in 5 parameters:
* **model** - optional: tiny / small / base / large. Default to tiny.
* **data** - default to data/mnist/test.
* **save_fp** - decide if to save false positive samples into sub-folder *results/false_positives*.
* **acc_thresh** - default to 0. If > 0, the early stopping procedure will be activated.
* **device** - default to 'cuda:0'.

Script *main.py* gives the following output:
* Plots for calibrated errors and confusion matrix are saved in *results*.
* ECE and MCE values are printed on-screen.
* False positive samples are stored in *results/false_positives*.
* False positive improvement results are printed on-screen. Print nothing if *false_positives* folder does not exist or is empty.

### Script Component Diagram

![code-structure-breakdown](img/code-structure-breakdown.png)
The script components can be divided into two blocks - **running** and **testing**. The order of import and usages are demonstrated in above diagram. 

**Note**: The 5 input params can be set alternatively with config.py as well as other hyper parameters.

<p align="right">(<a href="#table-of-content">back to top</a>)</p>

## Techniques Employed

### Early Stopping
Set parameter acc_thresh to value above **0** to enable this behavior.

```diff
- UserWarning: Model accuracy below set threshold. Terminating evaluation now...
```
Assume dataloader randomly shuffles, if accuracy for the first batch is lower than set threshold, the script will terminate batch iteration and return CE and save plots for this batch.

### Testing Logic
In case of any changes to the scripts or using scripts on new inputs, tests against functions are created to quickly examine the validity of function or compatibility between new input and the function to save debugging efforts. Testing scripts are saved in *testing* sub-folder.

The tests are to assert matching of set input-out pairs that are designed to handle different case scenarios (except plot tests which relies on manual check of plots). For example, 2 cases are covered for MCE, ECE tests - when some bins are of 0 count vs. all bins are of 0 count:
```python
    assert round(float(ece), 3) == 0.192, "wrong ECE for normal CE"
    assert round(float(mce), 3) == 0.390, "wrong mce for normal CE"

    assert get_mce(ce) == -0.1, "wrong MCE for all non-existent CE"
    assert get_ece(ce_b, batch_size) == -0.1, "wrong ECE for all non-existent CE"
```

For confusion matrix test, 3 scenarios are covered - when some prediction is correct / all are correct / prediction and label set is empty:
```python
    cm = get_confusion_matrix(y_pred, y_true, n_class=class_size)
    assert cm.tolist() == [[2, 2, 1],
                           [0, 2, 1],
                           [1, 0, 1]], "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_normal")

    cm = get_confusion_matrix(y_pred, y_pred, n_class=class_size)
    assert cm.tolist() == [[3, 0, 0],
                           [0, 4, 0],
                           [0, 0, 3]], "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_all_correct")

    cm = get_confusion_matrix(torch.tensor([]), torch.tensor([]), n_class=class_size)
    assert cm.tolist() == torch.zeros(class_size, class_size).int().tolist(), \
        "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_empty")
```

**Note**: Need to pass all tests before branch merge.


### False Positives Improvement
The idea is to reduce data dimensionality and perform clustering on the lower D dataset.

This script uses **T-sne** for d-reduction and **K-means** for clustering.

<p align="right">(<a href="#table-of-content">back to top</a>)</p>


## Limitation and Improvement

* ### Padding
  CovNext model doesn't work with MNIST dataset with a padding lower than 2. It returns this error:
  ```diff
  - RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (2 x 2). Kernel size can't be greater than actual input size
  ```
  Setting padding to 2 will fix it, but I haven't figured out why due to time limitation and being not familiar with ConvNext model. Adjusting padding automatically will generalize better to wider dataset.
* ### Better Early Stopping
  
  The current early stopping strategy relies on accuracy alone. This could be changed to taking {metric: threshold}, letting users define their own multi-threshold. Need to write {metric_name: metric_function} lookup for this to happen.
* ### More metrics

  For better evaluation and debugging potential, including more metrics and visualization of metrics will definitely improve usability. For example, for classification task, precision, recall, f1, accuracy, ROC and AUC can be included.


* ### Database Integration

  Currently, the evaluation outcomes are saved locally, meaning that rerun scripts unless change save path will write over past records. Integration with database, submitting session-id and metric values via database API will allow safer storage and past session lookup, and for visual comparison among sessions with self-made or out-of-shelf tools like tensorboard.


<p align="right">(<a href="#table-of-content">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install the required packages with pip to get started.
* pip
  ```sh
  pip install -r requirements.txt
  ```


<p align="right">(<a href="#table-of-content">back to top</a>)</p>


<!-- USAGE -->
## Usage

### Running
Can simply run main.py with parameters readily defined in **config.py**:
```commandline
python3 main.py
```
Otherwise, please define them with commandline:
```commandline
python3 main.py -d dataset/mnist/test -m base -t 0.2 -s true --device cpu
```

### Testing
To run individual script:
```commandline
python3 test_metrics.py
```
To run all:
```commandline
sh run.sh
```
The testing plots and false positives are saved in *testing/vis* and *testing/results/false_positives*
<p align="right">(<a href="#table-of-content">back to top</a>)</p>