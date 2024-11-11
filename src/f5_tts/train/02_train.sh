

python finetune_cli.py \
    --dataset_name "indo_f5" \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 1000 \
    --batch_size_type "frame" \
    --max_samples 64 \
    --grad_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --epochs 100 \
    --num_warmup_updates 200 \
    --save_per_updates 1000  \
    --last_per_steps 4000 \
    --tokenizer "custom" \
    --tokenizer_path "/home/ubuntu/F5-TTS/data/indo_f5/vocab.txt" \
    --logger "wandb" \
    --log_samples True \




    