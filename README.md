# Reproducing the Alignment Faking Paper of Deepseek-r1

This repository aims to reproduce the experiment from **Alignment Faking** paper by **Deepseek-r1**. You can access the paper [here](https://arxiv.org/abs/2412.14093). The project includes various scripts for running specific experiments and analyzing outputs. Below is an overview of the main components in the repository.

## Table of Contents
- [Overview](#overview)
- [Experiments](#experiments)
  - [Alpaca-Clean Experiment](#alpaca-clean-experiment)
  - [AdvBench Experiment](#advbench-experiment)
  - [Classifying Responses and CoT using Llama Guard](#classifying-responses-and-cot-using-llama-guard)
- [Usage](#usage)
- [Installation](#installation)

## Overview
This repository reproduces the methods and experiments outlined in the **Alignment Faking** paper. It includes scripts to perform the following tasks:
1. Running the Alpaca-Clean experiment.
2. Running the AdvBench experiment.
3. Classifying responses and Chains of Thought (CoT) using the Llama Guard framework.

## Experiments

### Alpaca-Clean Experiment
The `run.py` script is used for the **Alpaca-Clean** experiment. 

### AdvBench Experiment
The `run_harmful_behaviors.py` script is designed to run the **AdvBench** experiment. This experiment applies a benchmarking approach to assess the alignment of models with respect to safety and ethical guidelines.

### Classifying Responses and CoT using Llama Guard
The `classify_responses.py` script classifies AI responses and their associated Chains of Thought (CoT) using the **Llama Guard** model.

## Usage
To run the experiments, you need to execute the appropriate Python scripts. 

## Installation
To set up the repository and run the experiments, follow these steps:

1. **Clone the repository**  
   First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/susie-ku/test_deepseek.git
   cd test_deepseek

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
