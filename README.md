
# 🧬 VenusMutHub + TabPFN: Protein Mutation Effect Prediction with Foundation Models

This project investigates how transformer-based foundation models can predict the **functional impact of protein mutations** using **zero-shot tabular learning**. We apply [TabPFN](https://github.com/priorlabs/TabPFN) and its ensembling extensions to the [VenusMutHub dataset](https://huggingface.co/datasets/AI4Protein/VenusMutHub), a deep mutational scanning benchmark focused on the Venus fluorescent protein.

---

## 📌 Why This Matters

Protein mutation effect prediction is central to protein engineering, synthetic biology, and disease variant interpretation. Traditional models require tuning and large training sets, but TabPFN:
- Learns from millions of simulated tasks
- Makes **zero-shot predictions** in one pass
- Excels on **small, noisy biological datasets**

---

## 📚 Dataset: VenusMutHub (Hugging Face Datasets)

The [VenusMutHub dataset](https://huggingface.co/datasets/AI4Protein/VenusMutHub) provides deep mutational scanning data for the Venus fluorescent protein. Each entry represents a mutation, its altered sequence, and a fitness score corresponding to a biochemical property.

Each row includes:
- Protein mutation descriptors (e.g. position, amino acid change)
- A numerical `fitness_score` target

### How to Access the Dataset

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("AI4Protein/VenusMutHub")
```

➡️ [Read Hugging Face Datasets documentation](https://huggingface.co/docs/datasets)

---

## 🎯 Task & Evaluation

- **Goal:** Predict continuous mutation effects from protein descriptors
- **Input Features:** Encoded amino acid mutations (position, substitution)
- **Target:** `fitness_score` (e.g. fluorescence or stability)
- **Model:** TabPFNRegressor + AutoTabPFNRegressor
- **Evaluation Metrics:**
  - R² Score
  - Mean Squared Error (MSE)

---

## 🧠 Model: TabPFN

[TabPFN](https://github.com/priorlabs/TabPFN) is a transformer-based foundation model trained on millions of tabular tasks to support **zero-shot predictions** without hyperparameter tuning.

### Key Tools
- `TabPFNRegressor`: Predicts continuous labels on tabular data
- `AutoTabPFNRegressor`: From [TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions); uses post-hoc ensembling for improved accuracy

---

## 🧪 Results

After training on ~70% of the dataset and testing on the remaining 30%, we obtained the following:

| Model                 | Mean Squared Error | R² Score |
|----------------------|--------------------|----------|
| TabPFNRegressor       | 0.0294             | 0.7558   |
| AutoTabPFNRegressor   | 0.0293             | 0.7570   |

These results reflect **strong out-of-the-box generalization** for mutation effect prediction without domain-specific tuning.

---

## 🛠️ Setup & Installation

### 1. Clone this repository and create a virtual environment

```bash
git clone https://github.com/chewyuenrachael/tabpfn-venusmuthub.git
cd tabpfn-venusmuthub
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
tabpfn
datasets
pandas
scikit-learn
matplotlib
```

### 3. Install TabPFN Extensions (Optional, for ensembling)

```bash
git clone https://github.com/priorlabs/tabpfn-extensions.git
pip install -e tabpfn-extensions
```

---

## 🧪 Model Usage

```python
from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# Initialize and train
model = AutoTabPFNRegressor(device="cuda", max_time=60)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

---

## 📈 Results Summary

TabPFN demonstrates strong predictive performance with:

- **Minimal preprocessing**
- **No need for hyperparameter tuning**
- **Support for small/noisy biological datasets**

---

## 💡 Why Use TabPFN?

- 🧠 **Zero-shot predictions** with no model selection
- ⚡ **Fast inference** (seconds on GPU)
- 🧬 **Tailored for small-data bioinformatics problems**
- 🔍 SHAP-based interpretability extensions available

---

## 🧠 Future Directions

- Add **multi-target regression** for correlated protein properties
- Integrate **PLM embeddings** from models like ESM or ProtT5
- Explore **SHAP** and feature attribution from `tabpfn-extensions`
- Use `unsupervised` module for outlier detection in mutation sets

---

## 📜 Attributions

### Dataset
**AI4Protein. VenusMutHub**: A Benchmark for Protein Mutation Effect Prediction  
➡️ [Hugging Face Dataset](https://huggingface.co/datasets/AI4Protein/VenusMutHub)  
📜 Licensed under the MIT License

### Model
**TabPFN by Prior Labs**  
Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model*. Nature.  
[DOI: 10.1038/s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6)

**TabPFN Extensions**:  
[https://github.com/priorlabs/tabpfn-extensions](https://github.com/priorlabs/tabpfn-extensions)  
Apache 2.0 License

---

## 🙌 Acknowledgements

Built using tools from:

- 🤗 Hugging Face Datasets
- 🔬 Prior Labs' TabPFN ecosystem
- 🔥 PyTorch
- 📊 scikit-learn & matplotlib

Special thanks to the creators of **VenusMutHub** and **TabPFN** for providing robust tools for protein mutation analysis in modern machine learning pipelines.
