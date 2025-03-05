cd src || exit

python3 train.py casia-b ../data/casia-b_pose_train_valid.csv \
                 --valid_data_path ../data/casia-b_pose_test.csv \
                 --batch_size 64 \
                 --batch_size_validation 64 \
                 --embedding_layer_size 32 \
                 --layers 3\
                 --heads 4 \
                 --epochs 300 \
                 --learning_rate 1e-3 \
                 --temp 0.01 \
                 --network_name gt
