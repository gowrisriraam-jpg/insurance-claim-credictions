import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "insurance.csv")

df = pd.read_csv(csv_path)                                                          
