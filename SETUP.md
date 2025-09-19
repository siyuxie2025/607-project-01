# Project Setup Guide

Follow these steps to set up a clean environment and run this project.

---

## Install Python
Make sure you have **Python 3.13+** installed.

## Data Preparation
This project uses large datasets (>100MB) that are stored locally but not in Git.

### Raw Data
The data can be downloaded from [CreditCardFraud_updated](https://www.dropbox.com/scl/fi/amlnrhnefhwkyv6byn517/CreditCardFraud_updated.csv?rlkey=t6hnsl3w5c77xddtkxzjxasa1&st=fy22wrtx&dl=0). Place the data in `data/raw/CreditCardFraud_updated.csv`.


## Recreate Environment on Fresh System
```{bash}
git clone https://github.com/siyuxie2025/607-project-01
cd <your-project>

# create a virtual environment
python3 -m venv venv
source venv/bin/activate

# install the requirements
pip install -r requirements.txt

```