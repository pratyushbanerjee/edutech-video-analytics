# Edutech Video Analytics

A demonstration for detection of malpractice in videos using head pose estimation and gaze angle estimation

## Installation

Cloning the repository for the project

```bash
git clone https://github.com/pratyushbanerjee/edutech-video-analytics.git --recursive
```

Installing Deepgaze

```bash
git clone https://github.com/mpatacchiola/deepgaze.git
cd deepgaze
python setup.py install
```

### Setting up virtual environment (Optional)

Run the following commands in the project directory:

**For Linux**

```bash
python3.7 -m pip install --user virtualenv
python3.7 -m venv env
source env/bin/activate
```

**For Windows**

```bash
py -m pip install --user virtualenv
py -m venv env
.\env\Scripts\activate
```

Setting up kernel for Jupyter Notebook
```bash
pip install ipykernel
ipython kernel install --user --name=env
```

### Installing required modules

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
Run the following notebook to capture images and get frame statistics
```bash
jupyter notebook main.ipynb
```
Select the kernel 'env' from Kenrel > Change kernel  
Cell > Run All to run all the cells  
Press 'q' to stop capturing  
Captured images are stored in img_cap/  
Frame statistics are stored as outputs/img_stats.json  
  
Run this notebook to check for malpractice and reconstruct the video
```bash
jupyter notebook check.ipynb
```
Cell > Run All to run all the cells
Reconstructed video is stored in output/

### Python File

Run the following commands for the same using .py files
```bash
python main.py
python check.py
```