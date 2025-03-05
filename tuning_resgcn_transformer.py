import optuna
import os
import datetime
import wandb
import argparse

from train import main
from common import parse_option
from evaluate import _evaluate_casia_b


def objective(trial):
    opt = parse_option()
    
    # Define hyperparameter search space
    opt.embedding_layer_size = trial.suggest_categorical("embedding_layer_size", [64,128])
    opt.dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.05)
    opt.learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    opt.lr_decay_rate = trial.suggest_float("lr_decay_rate", 0.05, 0.2)
    opt.valid_data_path = '../data/casia-b_pose_test.csv'
    opt.batch_size = 32
    opt.epochs = 30
    opt.network_name = 'resgcn_transformer-n39-r8'
    opt.dataset = 'casia-b'
    # opt.train_data_path = '../data/casia-b_pose_test.csv'

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt.model_name = f"{date}_{opt.dataset}_{opt.network_name}" \
                     f"_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}"

    wandb.init(project='Gait_rec', name=opt.model_name, config=opt)

    opt.model_path = f"../save/{opt.dataset}_models"
    opt.tb_path = f"../save/{opt.dataset}_tensorboard/{opt.model_name}"

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.evaluation_fn = None
    if opt.dataset == "casia-b":
        opt.evaluation_fn = _evaluate_casia_b
    
    accuracy, loss = main(opt)  # Assume main() returns evaluation metrics

    return accuracy  # Optuna maximizes accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial)
