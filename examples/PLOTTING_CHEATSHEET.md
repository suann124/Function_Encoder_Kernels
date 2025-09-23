# 📊 Plotting Scripts Cheatsheet

## 🔧 **Script Overview**

### **1. `plot_experiments.py`** - Individual experiment plots
Single experiment analysis (progressive OR prune)

### **2. `plot_comparisons.py`** - Method & basis comparisons
Progressive VS prune comparisons (method + basis + variance)

### **3. `plot_degree_comparison.py`** - Polynomial degree analysis
D3 vs D4 vs D5 polynomial comparison

---

## 📋 **Common Arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `--save-dir DIR` | Output directory | `--save-dir plots` |
| `--save-as NAME` | Custom filename prefix | `--save-as "kepler_comparison"` |
| `--list` | List available experiments | `--list` |
| `--problem-type TYPE` | polynomial/kepler/vdp | `--problem-type kepler` |

---

## 1️⃣ **plot_experiments.py**

### Basic Usage
```bash
# Plot specific experiment
python plot_experiments.py examples/results/polynomial_progressive_d3

# With custom save location
python plot_experiments.py examples/results/kepler_prune_64 --save-dir my_plots
```

### Listing & Filtering
```bash
# List all experiments
python plot_experiments.py --list

# Filter by type
python plot_experiments.py --list --list-type polynomial
python plot_experiments.py --list --list-type progressive
python plot_experiments.py --list --list-type kepler
```

---

## 2️⃣ **plot_comparisons.py** ⭐

### Auto-Detection (Recommended)
```bash
# Find most recent polynomial experiments automatically
python plot_comparisons.py --problem-type polynomial

# Find most recent kepler experiments automatically
python plot_comparisons.py --problem-type kepler --save-dir plots

# Find any experiments with keyword filtering
python plot_comparisons.py --dataset 128 64 --save-as "kepler_128vs64"
```

### Manual Paths
```bash
# Specify exact experiment paths
python plot_comparisons.py \
  --progressive-path examples/results/kepler_progressive_dt01_20231201_120000 \
  --prune-path examples/results/kepler_prune_128 \
  --save-dir plots
```

### Comparison Types
```bash
# All comparisons (method + basis + variance) - DEFAULT
python plot_comparisons.py --problem-type polynomial --comparison-type all

# Only method comparison (eigenvalue spectra)
python plot_comparisons.py --problem-type kepler --comparison-type method

# Only basis comparison (2x4 streamplots/functions)
python plot_comparisons.py --problem-type kepler --comparison-type basis
```

### Dataset Keyword Search 🔍
```bash
# Search for experiments containing these keywords
python plot_comparisons.py --dataset dt01 128    # progressive=*dt01*, prune=*128*
python plot_comparisons.py --dataset d3 d3       # both contain "d3"
python plot_comparisons.py --dataset 64 128      # progressive=*64*, prune=*128*
```

### Complete Examples
```bash
# Polynomial comparison with custom naming
python plot_comparisons.py \
  --problem-type polynomial \
  --save-dir plots \
  --save-as "poly_method_comparison" \
  --comparison-type all

# Kepler streamplots only
python plot_comparisons.py \
  --problem-type kepler \
  --comparison-type basis \
  --dataset dt01 128 \
  --save-as "kepler_basis_dt01_vs_128"
```

---

## 3️⃣ **plot_degree_comparison.py**

### Auto-Detection
```bash
# Find D3, D4, D5 progressive experiments automatically
python plot_degree_comparison.py --method progressive

# Find D3, D4, D5 prune experiments automatically
python plot_degree_comparison.py --method prune --save-dir plots
```

### Manual Paths
```bash
# Specify exact paths for each degree
python plot_degree_comparison.py \
  --d3-path examples/results/polynomial_progressive_d3 \
  --d4-path examples/results/polynomial_progressive_d4 \
  --d5-path examples/results/polynomial_progressive_d5 \
  --save-as "polynomial_degrees_comparison"
```

### Listing
```bash
# See available degree experiments
python plot_degree_comparison.py --list
```

---

## 🎯 **Quick Reference - Most Common Commands**

### For Streamplots (Kepler Basis Functions):
```bash
python plot_comparisons.py --problem-type kepler --comparison-type basis
```

### For Method Comparison (Eigenvalue Spectra):
```bash
python plot_comparisons.py --problem-type polynomial --comparison-type method
```

### For Polynomial Degree Analysis:
```bash
python plot_degree_comparison.py --method progressive
```

### For Individual Experiment:
```bash
python plot_experiments.py examples/results/[experiment_name]
```

---

## 🔍 **Troubleshooting**

### "No experiments found"
1. Check available experiments: `python plot_comparisons.py --list`
2. Verify problem type: `--problem-type kepler/polynomial/vdp`
3. Use broader dataset keywords: `--dataset 128 64` instead of exact names

### Dataset keyword tips:
- Use partial matches: `128` instead of `kepler_prune_128_20231201_120000`
- Keywords are case-sensitive
- Both progressive and prune must exist with those keywords

### Output locations:
- Default: `plot_outputs/` directory
- Custom: `--save-dir your_directory`
- Multiple plots: Each gets a different filename when using `--comparison-type all`

---

## 📁 **Output Files**

### `plot_comparisons.py` outputs:
- `method_comparison.png` - Eigenvalue spectra comparison
- `cumulative_variance_comparison.png` - 99% threshold analysis
- `basis_comparison.png` - 2x4 basis functions/streamplots

### Filename patterns:
- With `--save-as "name"`: `name_method.png`, `name_basis.png`, etc.
- Default: Standard names as above