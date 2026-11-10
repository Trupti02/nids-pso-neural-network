from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PLOT_PATH = STATIC_DIR / "pso_convergence.png"

DOWNLOAD_BASE = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"

COLUMNS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
"urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
"num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
"dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]

# PSO settings
PSO_PARTICLES = 8
PSO_ITERATIONS = 10
PSO_W, PSO_C1, PSO_C2 = 0.7, 1.5, 1.5
FITNESS_SUBSET = 15000
FITNESS_MAX_ITER = 30
FINAL_MAX_ITER = 100

# Search space: [hidden neurons, log10(learning rate), log10(alpha)]
LOWER_BOUNDS = [8, -4, -6]
UPPER_BOUNDS = [128, -1, -2]

# Fields shown on the webpage
FORM_NUMERIC_FIELDS = ["duration","src_bytes","dst_bytes","count","srv_count","serror_rate",
    "rerror_rate","same_srv_rate","logged_in","num_failed_logins","dst_host_count","dst_host_srv_count"]