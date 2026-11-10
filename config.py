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

# SVM search space: [log10(C), log10(gamma)]
SVM_LOWER = [-1, -4]
SVM_UPPER = [3, 0]
SVM_FIT_TRAIN = 5000        # rows used to train each particle's SVM
SVM_FIT_VAL = 10000         # rows used to score each particle
SVM_FINAL_SAMPLES = 40000   # rows used for the final SVM (full data is very slow)

# Fields shown on the webpage
FORM_NUMERIC_FIELDS = ["duration","src_bytes","dst_bytes","count","srv_count","serror_rate",
    "rerror_rate","same_srv_rate","logged_in","num_failed_logins","dst_host_count","dst_host_srv_count"]