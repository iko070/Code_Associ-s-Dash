# 🎯 Multi-Objective Optimization & Interactive Dashboard

Welcome to this repository! This project provides a complete pipeline for **multi-objective optimization**, from initial modeling to algorithm tuning and final visualization via an interactive dashboard. Whether you're an engineer, data scientist, or optimizer at heart — there's something here for you.

## 🧠 Project Structure
- ```main.py```

The starting point of the project. This script sets up the **initial multi-objective optimization problem** using relevant models, objectives, and constraints.
    
  Think of it as your launchpad for exploration into the trade-offs of complex systems.
    
- ```Hyperparameters_tuning.py```

Fine-tunes the optimization algorithm (e.g., PSO, NSGA-II) by adjusting its 、**hyperparameters** to improve performance, convergence, and solution quality.

Because even the best algorithms need a little tuning to shine ✨

- ```Dashboard_Parameters```

This folder contains all files needed to display an **interactive HTML dashboard** for visualizing your optimization results, Pareto fronts, and parameter insights.

## 🚀 How to Use
1. **Install Dependencies**

Make sure you have the required Python packages installed:
```bash
pip install -r requirements.txt
```
2. **Run the Initial Model**
```bash
python main.py
```
This generates initial results for the multi-objective optimization problem.

3. **Optimize Hyperparameters**
```bash
python Hyperparameters_tuning.py
```
This will search for the best configuration of your optimizer for improved results.

4. **Launch the Dashboard**
- Download or clone the folder Dashboard_Parameters/ including all its contents.
- Locate and double-click index.html inside that folder.
- It will open in your **local web browser** and present an interactive dashboard.
- No server needed. Just click and explore 📊

## 🌟 Features
- ✅ Multi-objective optimization framework (e.g., minimize weight, maximize performance)
- 🔧 Easy-to-modify Python scripts for experimentation
- 📈 Beautiful local dashboard for intuitive result analysis
- 💡 Clear separation between modeling and hyperparameter tuning

## 📂 File Overview
| File/Folder                 | Description                                |
| --------------------------- | ------------------------------------------ |
| `main.py`                   | Initial multi-objective problem modeling   |
| `Hyperparameters_tuning.py` | Hyperparameter optimization for algorithms |
| `Dashboard_Parameters/`     | Local dashboard files (open `index.html`)  |

## 📝 License
This project is open source and available under the [MIT License](LICENSE).

Let data guide your decisions — and let this project be your compass. 🧭
