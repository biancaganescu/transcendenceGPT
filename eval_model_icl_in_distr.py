import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import numpy as np
import matplotlib.pyplot as plt
# PARAMS
temperature = 0.0001
top_k=5
seed = 1337
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
out_dir = None
BIG_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
SMALL_STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
exec(open('configurator.py').read()) 

torch.set_printoptions(precision=10, threshold=float('inf'))

# Torch configs
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

for steps in BIG_STEPS:
    print("\n")
    print(f"===================== EVALUATING FOR CHECKPOINT {steps} ======================")
    ckpt_path = os.path.join(out_dir, f'ckpt_step{steps}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)


    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

        # for shots in range(1, 30, 2):
        #     print("\n")
        #     print("TESTING FOR " + str(shots) + " SHOTS")
    test_data = torch.load("./data/in_distr_icl_test/seq_len_50.pt")[0]

    correct_predictions = 0

    SET_TOKEN = 3
    NO_SET_TOKEN = 4

    confusion_matrix = np.zeros((2, 3), dtype=int)

    for t in range(0, 50):
        confusion_matrix = np.zeros((2, 3), dtype=int)
        print("\n")
        print("For " + str(t) + " shots")
        # run generation
        with torch.no_grad():
            with ctx:
            
                num_samples = len(test_data)
                for i in range(num_samples):
                    input_sequence = test_data[i, :18 * t  + 16].unsqueeze(0).to(device)

                    actual_next_token = test_data[i, 18 * t  + 16]

                    
                    predicted_next_token = model.generate(input_sequence, 1, temperature=temperature, top_k=top_k)[0][-1]
                    
                    actual_idx = 1 if actual_next_token == SET_TOKEN else 0
                    
                    # Determine predicted label index:
                    # 0 for NO_SET, 1 for SET, and 2 for any "other" prediction.
                    if predicted_next_token == SET_TOKEN:
                        pred_idx = 1
                    elif predicted_next_token == NO_SET_TOKEN:
                        pred_idx = 0
                    else:
                        pred_idx = 2
                    
                    confusion_matrix[actual_idx, pred_idx] += 1


            tn = confusion_matrix[0, 0]  # actual NO_SET predicted as NO_SET
            fp = confusion_matrix[0, 1]  # actual NO_SET predicted as SET
            # For actual SET, both predictions as NO_SET (col 0) and as OTHER (col 2) are misclassifications:
            fn = confusion_matrix[1, 0] + confusion_matrix[1, 2]  # actual SET predicted as NO_SET or OTHER
            tp = confusion_matrix[1, 1]  # actual SET predicted as SET

            # Optionally, report breakdown of "other" predictions:
            other_no_set = confusion_matrix[0, 2]  # actual NO_SET predicted as OTHER
            other_set   = confusion_matrix[1, 2]  # actual SET predicted as OTHER

            # Calculate overall metrics:
            total = confusion_matrix.sum()
            accuracy = (tp + tn) / total * 100
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Print results:
            print("Confusion Matrix:")
            print(f"TN={tn} (Token {NO_SET_TOKEN} predicted as {NO_SET_TOKEN})")
            print(f"FP={fp} (Token {NO_SET_TOKEN} predicted as {SET_TOKEN})")
            print(f"Other predictions for NO_SET: {other_no_set} (Token {NO_SET_TOKEN} predicted as OTHER)")
            print(f"FN={fn} (Token {SET_TOKEN} predicted as NO_SET or OTHER)")
            print(f"TP={tp} (Token {SET_TOKEN} predicted as {SET_TOKEN})")
            print("Additional breakdown for SET:")
            print(f"Other predictions for SET: {other_set} (Token {SET_TOKEN} predicted as OTHER)")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        # # Plot confusion matrix
        # plt.figure(figsize=(8, 6))
        # plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        # plt.colorbar()
        # tick_marks = [0, 1]
        # plt.xticks(tick_marks, ["NO_SET", "SET"])
        # plt.yticks(tick_marks, ["NO_SET", "SET"])
        # plt.xlabel('Predicted Token')
        # plt.ylabel('Actual Token')

        # thresh = confusion_matrix.max() / 2
        # for i in range(2):
        #     for j in range(2):
        #         plt.text(j, i, format(confusion_matrix[i, j], 'd'),
        #                 horizontalalignment="center",
        #                 color="white" if confusion_matrix[i, j] > thresh else "black")

        # plt.tight_layout()
        # Uncomment below to save image on disk
        # plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
        # print(f"Confusion matrix saved to {os.path.join(out_dir, 'confusion_matrix.png')}")


