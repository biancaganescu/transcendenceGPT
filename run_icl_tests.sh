#!/bin/bash

python eval_model_icl_in_distr.py config/small_transcendence_gpt_50_train_50_test.py > results/small_model_icl_in_distr.txt

python eval_model_icl_in_distr.py config/big_transcendence_gpt_50_train_50_test.py > results/big_model_icl_in_distr.txt

python eval_model_icl_ood.py config/small_transcendence_gpt_50_train_50_test.py > results/small_model_icl_ood.txt

python eval_model_icl_ood.py config/big_transcendence_gpt_50_train_50_test.py > results/big_model_icl_ood.txt