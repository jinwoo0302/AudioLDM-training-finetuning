import argparse
import os
import shutil
import yaml
import torch 

from torch.utils.data import DataLoader

from audioldm_train.utilities.data.dataset import AudioDataset
from audioldm_train.utilities.model_util import instantiate_from_config
from audioldm_train.utilities.tools import copy_test_subset_data, get_restore_step

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # to prevent memory fragments

def main(configs, config_yaml_path, exp_group_name, exp_name):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)
    
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        ) # highest, high, medium

    log_path=configs["log_directory"]
    batch_size=configs["model"]["params"]["batchsize"]

    # additional data processing
    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    train_dataset=AudioDataset(configs, split="train", add_ons=dataloader_add_ons)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True, # enables faster data transfer from CPU to GPU
        shuffle=True,
        persistent_workers=True, #reuse workers without initialization
        prefetch_factor=2 # GPU waiting time decreases
    )

    print("The length of the dataset is %s,the length of the dataloader is %s, the batchsize is %s" 
          %(len(train_dataset),len(train_loader),batch_size))
    
    val_dataset=AudioDataset(configs, split="test", add_ons=dataloader_add_ons)
    val_loader=DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Copy test data
    test_data_subset_folder=os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        val_dataset.dataset_name
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset.data, test_data_subset_folder)

    try:
        config_reload_from_ckpt=configs["reload_from_chpt"]
    except:
        config_reload_from_ckpt=None
    
    try:
        limit_val_batches=configs["step"]["limit_val_batches"]
    except:
        limit_val_batches=None

    validation_every_n_epochs=configs["step"]["validation_every_n_epochs"]
    save_checkpoint_every_n_steps=configs["step"]["save_checkpoint_every_n_steps"]
    max_steps=configs["step"]["max_steps"]
    save_top_k=configs["step"]["save_top_k"]

    checkpoint_path=os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path=os.path.join(log_path,exp_group_name,exp_name)

    checkpoint_callback=ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    is_external_checkpoints=False
    if len(os.listdir(checkpoint_path))>0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path) # latest ckpt file name
        resume_from_checkpoint=os.path.join(checkpoint_path, restore_step) # latest ckpt path
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None: 
        resume_from_checkpoint=config_reload_from_ckpt
        is_external_checkpoints=True
        print("Reload ckpt specified in the config file", resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint=None

    devices=torch.cuda.device_count()

    latent_diffusion=instantiate_from_config(configs["model"]) # initialize an ldm with configs
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    wandb_logger=WandbLogger(
        save_dir=wandb_path,
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    latent_diffusion.test_data_subset_path=test_data_subset_folder

    print("Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("Perform validation every %s epochs" % validation_every_n_epochs)

    # torch.cuda.empty_cache() # memory fragments prevention

    trainer=Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
    )

    if is_external_checkpoints:
        if resume_from_checkpoint is not None:
            ckpt=torch.load(resume_from_checkpoint)["state_dict"]

            key_not_in_model_state_dict=[]
            size_mismatch_keys=[]
            state_dict=latent_diffusion.state_dict()

            print("Filtering key for reloading:", resume_from_checkpoint)
            print(
                "State dict key size:",
                len(list(state_dict.keys())), # model key
                len(list(ckpt.keys())) # checkpoint key
            )
            for key in tqdm(list(ckpt.keys())): # del keys in ckpt not contained in the ldm
                if key not in state_dict.keys():
                    key_not_in_model_state_dict.append(key)
                    del ckpt[key]
                    continue
                if state_dict[key].size()!=ckpt[key].size():
                    del ckpt[key]
                    size_mismatch_keys.append(key)
            
            latent_diffusion.load_state_dict(ckpt, strict=False)
            
        trainer.fit(latent_diffusion, train_loader, val_loader)
    else:
        trainer.fit(
            latent_diffusion, train_loader, val_loader, ckpt_path=resume_from_checkpoint
        )



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=False, 
        help="path to config .yaml file"
    )
    parser.add_argument(
        "--reload_from_ckpt", # vae in ldm
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )

    args=parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    
    config_yaml=args.config_yaml

    exp_name=os.path.basename(config_yaml.split(".")[0]) #config name
    exp_group_name=os.path.basename(os.path.dirname(config_yaml)) #config dir name

    config_yaml_path= config_yaml
    config_yaml=yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader) 

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"]=args.reload_from_ckpt


    main(config_yaml, config_yaml_path, exp_group_name, exp_name)