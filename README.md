# 2E-VRP-D Exact Algorithm Solver

> **A Python implementation of the Branch-and-Price exact algorithm for the Two-Echelon Vehicle Routing Problem with Drones (2E-VRP-D).**

This repository reproduces the exact algorithm proposed in:

> **Zhou, H., Qin, H., Cheng, C., & Rousseau, L.-M. (2023).**
> *An exact algorithm for the two-echelon vehicle routing problem with drones.*


---

## 📌 Problem Description

The **Two-Echelon Vehicle Routing Problem with Drones (2E-VRP-D)** considers a hybrid delivery system in which:

- A fleet of **trucks** (first echelon) departs from a central depot and visits customer locations.
- Each truck carries up to **Γ drones** (second echelon) that can be dispatched from truck locations to serve nearby drone-eligible customers.
- The objective is to **minimize total route duration** (travel time + waiting time) subject to vehicle capacity, drone energy, and time-window constraints.

---

## 🏗️ Repository Structure

```
.
├── main.py                  # Interactive CLI entry point
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── data/                    # Benchmark instances
│   ├── Cardiff10_01.txt     # 10-customer instances (×20)
│   ├── Cardiff15_01.txt     # 15-customer instances (×20)
│   ├── Cardiff25_01.txt     # 25-customer instances (×20)
│   └── c35_01.txt           # 35-customer instances (×9)
└── src/                     # Algorithm source code
    ├── MILP.py              # Compact MILP formulation (CPLEX)
    ├── branch_and_bound.py  # Branch-and-Bound tree manager
    ├── column_generation.py # Column Generation loop
    ├── rmp_solver.py        # Restricted Master Problem (Set Partitioning LP)
    ├── labeling.py          # Forward labeling (exact pricing)
    ├── bidirectional_labeling.py  # Bidirectional labeling (§4.1.1–4.1.3)
    ├── tabu_search.py       # Tabu Search heuristic pricing (§4.1.4)
    ├── initial_columns.py   # Initial column generation (cheapest insertion)
    ├── data_loader.py       # Cardiff benchmark data loader
    └── route.py             # Route data structure
```

---

## 🔑 Algorithm Overview

This implementation contains **two independent solvers**:

### 1. Branch-and-Price (B&P) — Exact Algorithm

The main solver faithfully reproduces the algorithm from the paper:

| Component | Description | Paper Reference |
|-----------|-------------|-----------------|
| **Set Partitioning RMP** | LP relaxation solved via CPLEX | §3, Eqs. (32)–(36) |
| **Forward Labeling** | ESPPDRC exact pricing | §4.1.1 |
| **Bidirectional Labeling** | Half-time label joining for efficiency | §4.1.1–4.1.3 |
| **Tabu Search Pricing** | Heuristic with Insertion/Removal/Shift operators | §4.1.4 |
| **4-Level Branching** | Vehicle count → Drone count → Arc flow → Drone dispatch | §4.3 |
| **Best-First B&B** | Priority queue by LP lower bound | §4.3 |

### 2. MILP Solver — Compact Mathematical Model

A direct MILP formulation with all decision variables and constraints as presented in §2 of the paper, solved by CPLEX as a reference model.

---

## ⚙️ Installation

### Prerequisites

This project requires **IBM ILOG CPLEX Optimization Studio 12.8+**, which is **not** available via pip.

- **Academic users**: obtain a free license at [IBM Academic Initiative](https://www.ibm.com/academic/technology/data-science)
- **Commercial users**: purchase a license from [IBM](https://www.ibm.com/products/ilog-cplex-optimization-studio)

### Step 1 — Install CPLEX

Install CPLEX 12.8 and set up the Python API:

```bash
# After installing CPLEX Studio, navigate to the python API directory:
cd /path/to/CPLEX_Studio128/cplex/python/<python-version>/<platform>/
python setup.py install
```

### Step 2 — Set Up the Python Environment

**Recommended (Conda):**

```bash
conda create -n cplex128_env python=3.6
conda activate cplex128_env
pip install -r requirements.txt
```

**Or with pip only** (after activating your environment):

```bash
pip install -r requirements.txt
```

### Step 3 — Verify CPLEX is Available

```python
python -c "import cplex; print(cplex.__version__)"
# Expected output: 12.8.0.0 (or similar)
```

---

## 🚀 Quick Start

### Interactive Mode (Recommended)

```bash
python main.py
```

You will be prompted to:
1. Choose an algorithm: **Branch-and-Price** or **MILP**
2. Choose a benchmark instance from the `data/` folder

### Command-Line Mode

**Run B&P solver directly:**

```bash
cd src
python branch_and_bound.py ../data/Cardiff10_01.txt
```

**Run MILP solver directly:**

```bash
cd src
python MILP.py ../data/Cardiff10_01.txt
```

**Run column generation only (for debugging):**

```bash
cd src
python column_generation.py ../data/Cardiff10_01.txt
```

---

## 📊 Benchmark Instances

The `data/` directory contains benchmark instances from the paper:

| Set | Customers | Instances | Notes |
|-----|-----------|-----------|-------|
| `Cardiff10` | 10 | 20 | Small, solvable optimally in seconds |
| `Cardiff15` | 15 | 20 | Medium |
| `Cardiff25` | 25 | 20 | Large |
| `c35` | 35 | 9  | Very large |

### Instance File Format

Each `.txt` file contains the following sections:
```
<n>          # Number of customers
<k_v>        # Number of trucks
<k_d>        # Total number of drones
<Gamma>      # Max drones per truck
<Q>          # Truck capacity
<q1_d>       # Drone + equipment weight
<q2_d>       # Drone body weight
<t_0>        # Drone loading/charging time
<vel_v>      # Truck speed
<vel_d>      # Drone speed
<B_c>        # Battery capacity
<Para>       # Energy consumption coefficient
<(n+1)×(n+1) distance matrix>
<customer info: index, x, y, drone_eligible, demand, time_window_ub, service_time_v, service_time_d> ...
```

---

## 🧪 Running Tests

```bash
cd src
python run_tests.py
```

Or test a specific Cardiff instance:

```bash
cd src
python test_cardiff.py
```

---

## 📐 Mathematical Formulation

The problem is defined over:

- **V = {0, 1, ..., n}**: nodes (depot 0 + customers)
- **Z = {1, ..., n}**: customer set
- **Z_d ⊆ Z**: drone-eligible customers
- **K_v**: truck fleet, **K_d**: drone fleet

**Objective**: Minimize total route duration (sum of travel and waiting times).

Key constraints enforce:
- Customer coverage (each customer served by exactly one truck or drone)
- Truck flow conservation and capacity
- Drone energy feasibility (`E[i,j] = 1` iff drone can reach j from i and return)
- Time-window deadlines for all customers
- Drone dispatching precedence and sequencing

The B&P approach decomposes this into a **Set Partitioning Master Problem** and independent **Pricing Subproblems** (one per drone configuration `k̄_d ∈ {0,...,Γ}`), solved via bidirectional labeling with dominance pruning.

---

## ⚠️ Known Limitations

- **CPLEX is required**: Both the MILP and B&P solvers use CPLEX as the LP/MIP engine. There is no open-source LP solver backend at this time.
- **Performance on large instances**: The B&P algorithm may be slow on `Cardiff25` and `c35` instances due to the exponential nature of the pricing problem.
- **Python version**: Tested with Python 3.6 and CPLEX 12.8. Compatibility with newer Python/CPLEX versions is not guaranteed.

---



## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

