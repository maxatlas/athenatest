<!-- TABLE OF CONTENTS -->
## Table of Content
<summary></summary>
<ol>
  <li>
    <a href="#about-the-project">Project Overview</a>
    <ul>
      <li><a href="#io-diagram">IO Diagram</a></li>
      <li><a href="#code-structure-breakdown">Script Component Diagram</a></li>
    </ul>
    </li>
  <li>
    <a href="#about-the-project">Techniques Employed</a>
    <ul>
      <li><a href="#io-diagram">Early Stopping</a></li>
      <li><a href="#io-diagram">Testing Logic</a></li>
      <li><a href="#io-diagram">False Positive Improvement</a></li>
    </ul>
  </li>
  <li>
      <a href="#io-diagram">Future Integration</a>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
    </ul>
  </li>
  <li><a href="#usage">Usage</a></li>
  <ul>
      <li><a href="#prerequisites">Running</a></li>
      <li><a href="#prerequisites">Testing</a></li>
  </ul>
</ol>

<!-- ABOUT THE PROJECT -->
## Project Overview
This is a python script to evaluate the performance of a classification model. The script is designed to help a fellow Data Scientist asses the performance of a model and potentially debug / or assure their model’s performance. 

### IO Diagram
![scriptIO](img/scriptIO.png)
The script *main.py* takes in 5 parameters:
* **model** - optional: tiny / small / base / large. Default to tiny.
* **data** - default to data/mnist/test.
* **save_fp** - decide if to save false positive samples into sub-folder *results/false_positives*.
* **acc_thresh** - default to 0. If > 0, the early stopping procedure will be activated.
* **device** - default to 'cuda:0'.

Script *main.py* gives output:
* Plots for calibrated errors and confusion matrix are saved in *results*.
* ECE and MCE values are printed on-screen.
* False positive samples are stored in *results/false_positives*.
* False positive improvement results are printed on-screen. Print nothing if *false_positives* folder does not exist or is empty.

### Script Component Diagram

![code-structure-breakdown](img/code-structure-breakdown.png)
The script components can be divided into two blocks - **running** and **testing**. The order of import and reasons are demonstrated in above diagram.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Techniques Employed

### Early Stopping



### Test script Logic
For easier code maintenance. 

### False Positives Improvement
Dimension reduction + some clustering techniques.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Future Integration
* More metrics

  For better evaluation and debugging potential, including more metrics to evaluate against will definitely help.


* Train
  
  Integrate with Training.


* Database API

  Record past outcomes of eval sessions.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get started, please install the required packages with pip.

### Prerequisites

Install the prerequisites with pip.
* pip
  ```sh
  pip install -r requirements.txt
  ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE -->
## Usage

### Running

### Testing


<p align="right">(<a href="#readme-top">back to top</a>)</p>