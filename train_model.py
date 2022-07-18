from model_training.trainer import Trainer
import json

glue_qqp_dir = "./data/QQP"
glove_vectors_path = "./data/glove.6B.50d.txt"

with open('configs/trainer_config.json') as json_file:
    config = json.load(json_file)

trainer = Trainer(
    glue_qqp_dir=glue_qqp_dir,
    glove_vectors_path=glove_vectors_path,
    path_to_save_weights=config["path_to_save_weights"],
    min_token_occurancies=config["min_token_occurancies"],
    random_seed=config["random_seed"],
    emb_rand_uni_bound=config["emb_rand_uni_bound"],
    freeze_knrm_embeddings=config["freeze_knrm_embeddings"],
    knrm_kernel_num=config["knrm_kernel_num"],
    knrm_out_mlp=config["knrm_out_mlp"],
    dataloader_bs=config["dataloader_bs"],
    train_lr=config["train_lr"],
    change_train_loader_ep=config["change_train_loader_ep"]
)

trainer.train(config["n_epochs"])
