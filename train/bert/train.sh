#!/usr/bin/env bash

python3 run_classifier.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=models/pre-trained/vocab.txt \
  --bert_config_file=models/pre-trained/bert_config.json \
  --output_dir=models/fine-tune/sim_model \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=models/pre-trained/bert_model.ckpt \
  --max_seq_length=50 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0
