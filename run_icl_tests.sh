#!/bin/bash

python eval_model_icl.py config/transcendence_gpt_2_train_2_test.py > results/results_icl_2_train.txt


python eval_model_icl.py config/transcendence_gpt_15_train_15_test.py > results/results_icl_15_train.txt


python eval_model_icl.py config/transcendence_gpt_30_train_30_test.py > results/results_icl_30_train.txt

