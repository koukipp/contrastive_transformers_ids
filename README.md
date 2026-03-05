 # Self-Supervised Transformer-based Contrastive Learning for Intrusion Detection Systems

This repository contains the source code for the model implementation and data processing pipeline described in the paper:
**[Self-Supervised Transformer-based Contrastive Learning for Intrusion Detection Systems](https://arxiv.org/abs/2505.08816)** presented at IFIP Networking 2025.

For the packet processing pipeline, we use a modified version of the **NFStream** framework, which you can find [here](https://github.com/koukipp/nfstream).

## Setup

Initialize your Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
apt install wireshark-common
```

## Data Preparation & Configuration

To train the model on benign traffic and evaluate it on malicious traffic, you must provide a flow-level labeling configuration for your `.pcap` files.

* **Example Configuration:** We provide an example configuration for the CICIDS2017 dataset in `cicids2017_config.json`.
* **Unlabeled Data:** If you do not provide a labeling configuration, all flows from the provided `.pcap` files will be considered benign. You will still be able to pretrain the model, but you will not be able to evaluate it.

## Usage

### 1. Full Pipeline (Preprocessing, Labeling, and Training)

To process raw `.pcap` files, label the flows, and train an instance of the model, run:

```bash
python train.py \
  --data-dir DATA_DIR_NAME \
  --labelled-data-dir LABEL_DATA_DIR_NAME \
  --labelling_config LABEL_CONFIG \
  --model_path MODEL_PATH_NAME

```

**Arguments:**

* `--data-dir`: Directory containing all the `.pcap` files you want to use.
* `--labelled-data-dir`: Target directory where the labeled packet dataset will be created.
* `--model_path`: The path where the trained model pipeline will be saved.

> **Note:** This step may take a while as it includes preprocessing the `.pcap` files (removing possible duplicate packets) and labeling flows.

### 2. Train on Existing Pre-processed Data

If you have already run the preprocessing step and your labeled data directory exists, you can skip data generation and jump straight to training:

```bash
python train.py \
  --labelled-data-dir LABEL_DATA_DIR_NAME \
  --model_path MODEL_PATH_NAME

```

### 3. Evaluate an Existing Model

To evaluate a previously trained model on labeled data, use the `--eval` flag:

```bash
python train.py --eval \
  --labelled-data-dir LABEL_DATA_DIR_NAME \
  --model_path MODEL_PATH_NAME

```

### 4. Real-Time Deployment

You can deploy the trained model for real-time intrusion detection on a live network interface:

```bash
python read_if.py IFACE_NAME MODEL_PATH_NAME

```

* `IFACE_NAME`: The network interface to monitor (e.g., `eth0`, `wlan0`).

---