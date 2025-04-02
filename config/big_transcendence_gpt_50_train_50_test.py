out_dir = 'out-transcendence-gpt-50-train-50-test'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'transcendence-gpt'
wandb_run_name = '50_train_50_test'

dataset = 'card_set_50x50_big'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 900 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0

learning_rate = 2e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

vocab_size = 17

# # on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

dtype = "float32"