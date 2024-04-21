dataset = 'CurveLanes'
data_root= '/kaggle/input/curvelanes/Curvelanes' # Need to be modified before running
epoch = 50
batch_size = 4
optimizer = 'SGD'
learning_rate = 0.003125
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695
use_aux = False
backbone = '34'
sim_loss_w = 1.0
shp_loss_w = 0.0
note = ''
log_path = ''
finetune = None
resume = None
test_model = ''
test_work_dir = ''
tta = False
num_lanes = 10
var_loss_power = 2.0
auto_backup = True
num_row = 72
num_col = 81
train_width = 1600
train_height = 800
num_cell_row = 200
num_cell_col = 100
mean_loss_w = 0.05
crop_ratio = 0.8
