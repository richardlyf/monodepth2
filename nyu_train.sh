python train.py --model_name mono_model --split nyu --dataset nyu --data_path /home/jupyter/nyu_dataset --batch_size 6 --num_epochs 5 --load_weights_folder ~/models/mono_640x192


python train.py --model_name mono_model --data_path /home/jupyter/nyu_dataset/ --log_dir ~/nyu --batch_size 6 --split nyu --dataset nyu --num_epochs 4 --width 256 --height 192 --load_weights_folder ~/models/mono_640x192/ --models_to_load encoder depth pose_encoder --max_train_size 5000 --max_val_size 500 --learning_rate 1e-5


# New commands
# og nyu
python train.py --model_name nyu_5epoch --split nyu --dataset nyu --data_path /home/jupyter/nyu_dataset/ --num_epochs 5 --width 256 --max_depth 10
# feature nyu
python train.py --model_name feate5_nyu_5epoch --split nyu --dataset nyu --data_path /home/jupyter/nyu_dataset/ --num_epochs 4 --width 256 --max_depth 10 --use_feature --autoencoder_pretrained_path /home/jupyter/feat_log/latest.pth


# Eval
python evaluate_depth_nyu.py --load_weights_folder ~/dataset/log/nyu_5epoch/models/weights_4 --eval_split nyu_labeled --data_path /home/jupyter/nyu_labeled/ --width 256 --max_depth 10 --eval_mono
