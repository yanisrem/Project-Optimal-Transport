# Optimal Transport Project Project: Benchmark Entropic Estimator vs. Other More Complex Alternatives (ICNN, Flow Matching)

Project carried out as part of the Optimal Transport course (ENSAE, 2024).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

## Introduction

We are evaluate the statistical and computational performance of the entropic map, ICNN-based approaches, and conditional flow matching.

## Features

- `experiment_simul_data.ipynb`: experiment on generated samples
- `experiment_caltech_amazon.ipynb`: Domain Adaptation: Caltech and Amazon images
- `experiment_digits.ipynb`: Domain Adaptation: MNIST and USPS images
- `src`: includes ICNN, OT solver and Mini-batches OT

## Getting Started

### Prerequisites

- Python (>=3.6)
- Other dependencies (specified in `requirements.txt`)

### Installation

```bash
git clone https://github.com/yanisrem/NLP-Project
cd src
pip install -r requirements.txt
```
