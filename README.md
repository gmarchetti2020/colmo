# COLMO
## A Llama-like model for educational purposes.
This code implements a decoder-only large(-ish, about 1B parameters) language model that includes several Llama-like advances:
- Rotary positional embeddings applied to the key and query tensors,
- Grouped Query Attention,
- SwiGLU activation function for the intermediate feed-forward network,
- RMS Normalization.

We provide a sample script to pre-train the model on a subset of Ai2's Dolma dataset, using 64 TPU v5e cores organized in a 16-host x 4-core
layout. 
We also provide a sample notebook for experimentation with a smaller implementation of the same architecture,
both in terms of model size and training dataset.
The notebook will work on a machine with 8 TPU cores (e.g. a Kaggle TPU VM) and includes further experiments with fine-tuning and generation
with different samplers.
The code uses the Keras library for portability and relies on the JAX framework for distributed computation.
## Instructions
1. Deploy a TPU cluster as per documentation, but do NOT run a startup script yet: https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm
2. Wait for about 1 hour - this is not included in the official documentation. Soon after the deployment an unattended upgrade process starts. Let it complete before installing anything else.
3. Edit the start-tpu-job-64.sh shell script to match your configuration (e.g. with the relative tpu cluster names, project id, staging bucket etc...)
5. Edit the config.ini file to match your desired configuration. As it is, it will produce a 1B-parameter model trained on the Dolma 1.6 sample dataset (about 10B tokens).
6. Copy the vocabulary file c4vocab.pkl, the config.ini file and the python code colmo2-tpu-pretrain-64.py to a staging bucket on GCS.
7. On the GCP Console, open a cloud shell and upload the start-tpu-job-64.sh script.
8. Either run the shell script line by line, to verify each step, or just run all of it in one pass.

**Note:** The cloud shell will time out if not used, depending on policies in your GCP organization.
We suggest running the last line of the script, which launches the computation, either on a node of the cluster directly, or on your machine,
if you have installed the gcloud utility there.  

You can monitor the progress of your computations by starting a tensorboard instance and pointing it to the gs://staging_bucket/logs/colmo2-tpu directory.
The easiest way to do so is to run tensorboard in a Colab notebook, where you can then run the following code:
```
!pip install -U tensorboard-plugin-profile
%load_ext tensorboard
%tensorboard --logdir gs://<your staging bucket>/logs/colmo2-tpu
```
-----
```
/*
 * THIS IS SAMPLE CODE
 *
 * Copyright 2025 Google
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```
