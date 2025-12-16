# EC Number Prediction using ESM2 Embeddings

This project implements multiple machine learning models for predicting Enzyme Commission (EC) numbers from protein sequences using ESM2 (Evolutionary Scale Modeling 2) embeddings. The project includes RandomForest, XGBoost, MLP (Multi-Layer Perceptron), and BiLSTM models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Processing and Feature Extraction](#1-data-processing-and-feature-extraction)
  - [Training Models](#2-training-models)
  - [Inference/Prediction](#3-inferenceprediction)
- [Model Performance](#model-performance)
- [File Descriptions](#file-descriptions)

## Overview

This project uses ESM2 (facebook/esm2_t12_35M_UR50D) to generate protein sequence embeddings, which are then used as features for various machine learning classifiers to predict EC numbers. The models are trained on a dataset of 236,607 protein sequences with 263 unique EC number classes.

> **📦 Pre-trained Models Available**: All trained models and generated embeddings are available on Google Drive for quick inference without training. See [Quick Start](#quic
> k-start) section for download instructions.

## Features

- **ESM2 Embedding Extraction**: Uses pre-trained ESM2 model to generate 480-dimensional embeddings
- **Multiple ML Models**:
  - RandomForest Classifier
  - XGBoost Classifier
  - Multi-Layer Perceptron (MLP)
  - Bidirectional LSTM (BiLSTM)
- **Standalone Inference**: Each model can be used independently for prediction
- **Top-K Predictions**: Returns top-k most likely EC numbers with confidence scores

## Requirements

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training and inference)
- At least 8GB RAM (16GB recommended)
- Sufficient disk space for models and embeddings (~2GB)

### Python Dependencies

All required packages are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

**Complete Dependency List**:

- **pandas** (>=1.5.0): Data manipulation and analysis
- **numpy** (>=1.23.0): Numerical computing
- **torch** (>=2.0.0): PyTorch deep learning framework (CUDA support recommended)
- **transformers** (>=4.30.0): Hugging Face transformers for ESM2 model
- **scikit-learn** (>=1.2.0): Machine learning utilities (RandomForest, LabelEncoder, etc.)
- **xgboost** (>=2.0.0): Gradient boosting framework
- **joblib** (>=1.2.0): Model serialization
- **tqdm** (>=4.65.0): Progress bars
- **ipywidgets** (>=8.0.0): Interactive widgets for Jupyter
- **jupyter** (>=1.0.0): Jupyter notebook environment (optional but recommended)
- **notebook** (>=6.5.0): Jupyter notebook server (optional but recommended)

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Dataset

### Dataset Information

- **File**: `3_levels_EC.tsv`
- **Format**: Tab-separated values (TSV)
- **Columns**:
  - `Entry`: Protein entry identifier
  - `Entry Name`: Protein entry name
  - `Protein names`: Full protein names
  - `Sequence`: Amino acid sequence (single-letter code)
  - `EC number`: Enzyme Commission number (target label)

### Dataset Statistics

- **Total samples**: 236,607
- **Unique EC classes**: 263
- **Train/Test split**: 80/20 (189,285 training, 47,322 testing)

### Dataset Source

The dataset should be placed in the project root directory as `3_levels_EC.tsv`. 

#### Obtaining the Dataset

The dataset can be obtained from the following sources:

1. **UniProt Database**:
   - Main website: https://www.uniprot.org/
   - UniProtKB search: https://www.uniprot.org/uniprotkb
   - Download page: https://www.uniprot.org/downloads
   - To filter for enzymes with EC numbers, use the query: `ec:*` in the search

2. **BRENDA Enzyme Database**:
   - Main website: https://www.brenda-enzymes.org/
   - Database access: https://www.brenda-enzymes.org/search_result.php
   - Provides comprehensive EC number annotations

3. **UniProt REST API** (for programmatic access):
   ```python
   # Example: Download proteins with EC numbers
   import requests
   url = "https://rest.uniprot.org/uniprotkb/search"
   params = {
       "query": "ec:*",
       "format": "tsv",
       "fields": "accession,id,protein_name,sequence,ec"
   }
   ```

4. **Direct Download Links** (if available):
   - UniProt releases: https://www.uniprot.org/downloads
   - Select "UniProtKB" → "Reviewed (Swiss-Prot)" or "Unreviewed (TrEMBL)"
   - Filter by EC numbers during download

#### Dataset Format Requirements

The dataset file `3_levels_EC.tsv` must have the following structure:

```
Entry	Entry Name	Protein names	Sequence	EC number
P12345	PROT1	Protein 1	MKTAYIAKQR...	1.1.1.1
P67890	PROT2	Protein 2	MKTAYIAKQR...	2.7.1.1
...
```

**Important Notes**:
- File must be tab-separated (TSV format)
- Column names must match exactly (case-sensitive)
- EC numbers should be in format: `X.X.X.X` or `X.X.X` (3-level EC numbers)
- Sequences should be in single-letter amino acid code (no spaces)
- Each row should have a valid EC number (no missing values in EC number column)

#### Dataset Preparation

If you need to create the dataset from UniProt:

1. Go to https://www.uniprot.org/uniprotkb
2. Search for: `ec:* AND reviewed:true` (for Swiss-Prot) or `ec:*` (for all)
3. Click "Download" → Select "TSV" format
4. Choose columns: Entry, Entry Name, Protein names, Sequence, EC number
5. Save as `3_levels_EC.tsv` in the project root directory

**Note**: The exact dataset used in this project may be a curated subset. Please ensure your dataset follows the same format with the columns listed above.

## Project Structure

### Files in GitHub Repository

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── 3_levels_EC.tsv                   # Dataset file
├── ML & Data Processing.ipynb        # Data processing, RF, and XGBoost training
├── MLP.ipynb                         # MLP model training and inference
└── BiLSTM.ipynb                      # BiLSTM model training and inference
```

### Pre-trained Models and Generated Files (Google Drive)

The following files are available on Google Drive to avoid re-running training:

- `esm2_features.npy` - Generated ESM2 embeddings (236,607 × 480)
- `rf_ec_esm2.joblib` - Trained RandomForest model
- `xgboost_ec_cpu_hist.json` - Trained XGBoost model
- `mlp_ec_esm2.pt` - Trained MLP model weights
- `bilstm_ec_esm2.pt` - Trained BiLSTM model weights
- `label_encoder_ec_esm2.joblib` - Label encoder for EC numbers

**Download Link**: [Google Drive - Pre-trained Models](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)

**Note**: Replace `YOUR_GOOGLE_DRIVE_LINK_HERE` with your actual Google Drive link. Make sure the link is set to "Anyone with the link can view" for public access.

## Quick Start

### Option 1: Using Pre-trained Models (Recommended)

For quick testing without training:

1. **Download pre-trained models from Google Drive**:
   - Go to: [Google Drive - Pre-trained Models](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)
   - Download all files to the project root directory:
     - `esm2_features.npy`
     - `rf_ec_esm2.joblib`
     - `xgboost_ec_cpu_hist.json`
     - `mlp_ec_esm2.pt`
     - `bilstm_ec_esm2.pt`
     - `label_encoder_ec_esm2.joblib`

2. **Ensure dataset is available**:
   - Place `3_levels_EC.tsv` in the project root directory

3. **Open any inference notebook**:
   - `MLP.ipynb` (Cells 2-6)
   - `BiLSTM.ipynb` (Cells 1-4)
   - `ML & Data Processing.ipynb` (Cells 10-14)

4. **Run the setup cells**, then use the prediction function with a protein sequence

### Option 2: Training from Scratch

If you want to train the models yourself, follow the full usage instructions below.

## Usage

### 1. Data Processing and Feature Extraction

**Notebook**: `ML & Data Processing.ipynb`

This notebook performs:
- Loading and preprocessing the dataset
- Generating ESM2 embeddings for all protein sequences
- Training RandomForest and XGBoost models
- Saving models and label encoder

**Steps**:
1. Open `ML & Data Processing.ipynb` in Jupyter Notebook/Lab
2. Run all cells sequentially:
   - Cell 0: Load dataset and setup
   - Cell 1: Load ESM2 model
   - Cell 2: Define embedding function
   - Cell 3: Generate embeddings (this may take several hours depending on GPU)
   - Cell 4: Train RandomForest model
   - Cell 6: Train XGBoost model
   - Cell 7: Generate classification report for XGBoost
   - Cell 8: Save XGBoost model

**Output files** (will be saved locally, or download from Google Drive):
- `esm2_features.npy`: ESM2 embeddings (236,607 × 480) - **~450 MB**
- `rf_ec_esm2.joblib`: Trained RandomForest model - **~500 MB**
- `label_encoder_ec_esm2.joblib`: Label encoder - **~50 KB**
- `xgboost_ec_cpu_hist.json`: Trained XGBoost model - **~200 MB**

**Note**: These files are also available for download from [Google Drive](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing) to save time.

### 2. Training Models

#### MLP Model

**Notebook**: `MLP.ipynb`

1. Open `MLP.ipynb`
2. Run Cell 0 to train the MLP model
3. The model will be saved as `mlp_ec_esm2.pt` (~50 MB)

**Alternative**: Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)

#### BiLSTM Model

**Notebook**: `BiLSTM.ipynb`

1. Open `BiLSTM.ipynb`
2. Run Cell 0 to train the BiLSTM model
3. The model will be saved as `bilstm_ec_esm2.pt` (~5 MB)

**Alternative**: Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)

**Note**: Both MLP and BiLSTM require `esm2_features.npy` to be generated first (or downloaded from Google Drive).

### 3. Inference/Prediction

Each notebook includes inference code that can be run independently:

#### RandomForest & XGBoost Inference

**Notebook**: `ML & Data Processing.ipynb` (Cells 10-14)

1. Run Cell 10: Load models and label encoder
2. Run Cell 11: Load ESM2 model
3. Run Cell 12: Define prediction functions
4. Run Cell 13 or 14: Test with a protein sequence

**Example**:
```python
# After running setup cells (10-12)
TEST_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

# For RandomForest
pred_ec_rf, conf_rf = predict_ec_rf(TEST_SEQ, top_k=3)

# For XGBoost
pred_ec_xgb, conf_xgb = predict_ec_xgb(TEST_SEQ, top_k=3)
```

#### MLP Inference

**Notebook**: `MLP.ipynb` (Cells 2-6)

1. Run Cell 2: Load model architecture and weights
2. Run Cell 3: Load model weights
3. Run Cell 4: Load ESM2 model
4. Run Cell 5: Define prediction functions
5. Run Cell 6: Test with a protein sequence

#### BiLSTM Inference

**Notebook**: `BiLSTM.ipynb` (Cells 1-4)

1. Run Cell 1: Load model and weights
2. Run Cell 2: Load ESM2 model
3. Run Cell 3: Define prediction functions
4. Run Cell 4: Test with a protein sequence

## Model Performance

Based on test set evaluation (20% holdout):

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| RandomForest | ~96.6% | 300 trees, trained incrementally |
| XGBoost | ~96.3% | 300 rounds, CPU histogram method |
| MLP | ~97.6% | 2 hidden layers (512 units each), dropout 0.3 |
| BiLSTM | ~95.8% | 2 layers, bidirectional, hidden size 128 |

**Note**: Actual performance may vary depending on the dataset and training conditions.

## File Descriptions

### Notebooks

- **ML & Data Processing.ipynb**: 
  - Data loading and preprocessing
  - ESM2 embedding generation
  - RandomForest and XGBoost training
  - Inference code for both models

- **MLP.ipynb**: 
  - MLP model definition and training
  - Standalone inference code

- **BiLSTM.ipynb**: 
  - BiLSTM model definition and training
  - Standalone inference code

### Model Files (Available on Google Drive)

All pre-trained models and generated files can be downloaded from:
**[Google Drive - Pre-trained Models](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)**

- **esm2_features.npy**: Pre-computed ESM2 embeddings (236,607 × 480) - ~450 MB
- **rf_ec_esm2.joblib**: Trained RandomForest classifier - ~500 MB
- **xgboost_ec_cpu_hist.json**: Trained XGBoost model - ~200 MB
- **mlp_ec_esm2.pt**: MLP model weights (PyTorch) - ~50 MB
- **bilstm_ec_esm2.pt**: BiLSTM model weights (PyTorch) - ~5 MB
- **label_encoder_ec_esm2.joblib**: Label encoder mapping EC numbers to integers - ~50 KB

**Total size**: ~1.2 GB

**Note**: These files are not included in the GitHub repository to keep it lightweight. Download them from Google Drive for quick inference without training.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in embedding generation (Cell 3 of ML & Data Processing.ipynb)
   - Use CPU for inference if GPU memory is limited

2. **Model files not found**:
   - **Option 1**: Download pre-trained models from [Google Drive](https://drive.google.com/file/d/1G4_dDKcvTQw6fyubDucJFcjfdBt_lxPg/view?usp=sharing)
   - **Option 2**: Run the training cells first to generate model files
   - Check that model files are in the project root directory
   - Verify file names match exactly (case-sensitive)

3. **ESM2 model download issues**:
   - The model will be downloaded automatically on first use
   - Ensure stable internet connection
   - Model size: ~140MB

4. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+)



