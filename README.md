# LeCasa

**LeCasa** is a Real2Sim2Real platform for **joint-level teleoperation, dataset collection, Vision-Language-Action (VLA) policy training, and evaluation** in both simulation and real robots.  
This work focuses on single- and multi-stage tasks in **RoboCasa kitchen environments** using the **SO101 arm**, with policies trained through the **LeRobot framework**.

---

## Installation

LeCasa works across all major computing platforms. The easiest setup is via **Anaconda**.

### 1. Create a conda environment
```bash
conda create -c conda-forge -n lecasa python=3.10
conda activate lecasa
```

### 2. Clone the repository
```bash
git clone https://github.com/sanan222/LeCasa.git
cd LeCasa/src/
```

### 3. Install RoboCasa environments

First, clone and install **robosuite**:
```bash
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .
```

Then, clone and install **RoboCasa**:
```bash
cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
```

⚠️ **Note:**  
The latest RoboCasa depends on the master branch of robosuite. If you want the legacy **RoboCasa v0.1**, use the `robocasa_v0.1` branch of robosuite.

---

If you encounter issues with `numpy` or `numba`, try installing via conda:

```bash
conda install -c numba numba=0.56.4 -y
```

---

Download assets and set up macros

```bash
python robocasa/scripts/download_kitchen_assets.py   # (~5GB download)
python robocasa/scripts/setup_macros.py
```

---

### 4. Install LeRobot
```bash
pip install lerobot
```

---

## Running simulation

To start data collection:

```bash
cd LeCasa/src/
python main.py
```

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Contributing

Contributions are welcome!  
Please feel free to open issues or submit pull requests.
