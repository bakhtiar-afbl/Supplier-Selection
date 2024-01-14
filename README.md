# Supplier Selection and Ranking

This Python script performs supplier selection and ranking based on various criteria, including pricing, delivery time, and historical performance.

## Overview

The script connects to a SQL Server database, retrieves supplier and procurement information, and uses machine learning models to rank suppliers. The ranking is based on factors such as base price, delivery time, order quantity, and recency of transactions.

## Requirements

- Python 3.x
- Required Python packages (install using `pip install package_name`):
  - pandas
  - numpy
  - scikit-learn
  - statsmodels
  - matplotlib
  - seaborn
  - pyodbc
  - cryptography

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Supplier-Selection.git


## Navigate to the project directory:

    cd Supplier-Selection

## Install the required Python packages:
    pip install -r requirements.txt

## Configuration

#### Update the script with your SQL Server connection details:
#### Open SupplierSelection.py and locate the following lines:

    server = input("Please enter SQL Server address: ") 

    encrypted_password = input("Please enter DB Password: ")

## Run the script:

python SupplierSelection.py

## Models Used
The script employs the following machine learning models:

    Random Forest Regressor
    Linear Regression
    Decision Tree Regressor
    Gradient Boosting Regressor
## Output
    The script outputs supplier rankings and relevant information.  The results are stored in a SQL Server table named pre.tblSupplierRank.

## License
This project is licensed under the MIT License.

Feel free to customize the README file based on your specific needs and project details. Include additional sections if necessary, such as contributing guidelines, acknowledgments, or troubleshooting tips.
