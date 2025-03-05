cd src || exit

python3 train.py casia-b ../data/casia-b_pose_train_valid.csv \
                 --valid_data_path ../data/casia-b_pose_test.csv \
                 --batch_size 64 \
                 --batch_size_validation 256 \
                 --embedding_layer_size 128 \
                 --epochs 300 \
                 --layers 3\
                 --heads 4 \
                 --learning_rate 4e-3 \
                 --temp 0.01 \
                 --network_name resgcn_transformer-n39-r8