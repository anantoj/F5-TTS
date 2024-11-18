

python finetune_cli.py \
    --dataset_name "id-en-f5" \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 6000 \
    --batch_size_type "frame" \
    --max_samples 64 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --epochs 22 \
    --num_warmup_updates 300 \
    --save_per_updates 10000 \
    --last_per_steps 100000 \
    --tokenizer "custom" \
    --tokenizer_path "/home/ubuntu/F5-TTS/data/id-en-f5/vocab.txt" \
    --logger "tensorboard" \
    --log_samples True \


# TODO: log tensorboard