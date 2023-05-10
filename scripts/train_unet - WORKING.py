
import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Optional
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (DDIMScheduler, DDPMScheduler,
                        UNet2DModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.training_utils import EMAModel
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm
from PIL import Image

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sonicdiffusion.pipeline_sonic_diffusion import AudioDiffusionPipeline

logger = get_logger(__name__) #start logger instance with current module name

# START MAIN FUNCTION

def main(args):
    
    """Runs the main training loop.

    Args:
        args: A namespace containing the command line arguments.

    Returns:
        None.

    Raises:
        NotImplementedError: If the accelerator type is not supported.
    """
    
    
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        logging_dir=logging_dir,
    )
 
    def load_dataset_or_disk(args):
        """
        Load a dataset from disk or remote repository based on the provided arguments.
        Args:
            args (Namespace): A namespace object containing the following attributes:
                dataset_name (str): Name of the dataset to be loaded.
                dataset_config_name (str): Configuration name for the dataset.
                cache_dir (str): Directory where the dataset should be cached.
                use_auth_token (bool): Whether to use an authentication token for loading private datasets.
                train_data_dir (str): Directory containing the training data when using the "imagefolder" dataset.

        Returns:
            dataset (Dataset): The loaded dataset object.
        """

        if args.dataset_name is not None:
            if os.path.exists(args.dataset_name):
                return load_from_disk(args.dataset_name, storage_options=args.dataset_config_name)["train"]
            else:
                return load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
            split="train",
            )
        else:
            return load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )

    dataset = load_dataset_or_disk(args)
  
    # Determine image resolution
    resolution = dataset[0]["image"].height, dataset[0]["image"].width

    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])

    def transforms(examples):
        images = [augmentations(image) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # Initialise logging 
    
    if accelerator.is_main_process:
        #Initialise WANDB
        accelerator.init_trackers(
            project_name=args.project_name,
            config={"num_epochs": args.num_epochs, 
                    "learning_rate": args.learning_rate,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "lr_scheduler": args.lr_scheduler,
                    "warm_up_steps": args.warmup_steps,
                    "adam_beta1":args.adam_beta1,
                    "adam_beta2":args.adam_beta2,
                    "adam_weight_decay": args.adam_weight_decay,
                    "ema_max_decay": args.ema_max_decay,
                    "ema_power": args.ema_power,
                    "ema_inv_gamma": args.ema_inv_gamma,
                    },
            init_kwargs={"wandb": {"resume": True}}
        )
        wandb_table_image = wandb.Table(
                    columns=['Epoch', 'Step', 'Clean-Images', 'Generated-Mel-Images'])
        
        wandb_table_audio = wandb.Table(
                    columns=['Epoch', 'Step', 'Generated-Audio'])
    
#________________ SELECT MODEL __________________________________
    if args.from_pretrained is not None:
        # Specify the name of the artifact (model)
        artifact_name = args.from_pretrained  
        artifact = wandb.use_artifact(artifact_name)

        # Download the model file(s) and return the path to the downloaded artifact
        artifact_dir = artifact.download()

        pipeline = AudioDiffusionPipeline.from_pretrained(artifact_dir)
        mel = pipeline.mel
        model = pipeline.unet

    else:
        model = UNet2DModel(
            sample_size=resolution,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
#________________ END SELECT MODEL __________________________________

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )


    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    global_step = 0
    
#________________ START TRAINING LOOP __________________________________
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz, ),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if ((epoch + 1) % args.save_model_epochs == 0
                    or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = AudioDiffusionPipeline(
                    vqvae=None,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )

            if (epoch + 1) % args.save_images_epochs == 0:
                generator = torch.Generator(
                    device=clean_images.device).manual_seed(42)


                # run pipeline in inference (sample random noise and denoise)
                images, (sample_rate, audios) = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    return_dict=False,
                )
                
                # denormalize the images and save to weights and biases
                # creates list of images in numpy format
                images = np.array([
                    np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                        (len(image.getbands()), image.height, image.width))
                    for image in images
                ])
                # Log images for epoch
                
                for img in images:
                    img_shape = np.reshape(img, (1, 256, 256))
                    wandb_table_image.add_data(
                        epoch, 
                        global_step, wandb.Image(clean_images[0]),
                        wandb.Image(img_shape))
                
                #log audio files
                
                wandb_table_audio.add_data(
                epoch,
                global_step,
                wandb.Audio(normalize(audios[0]), sample_rate=sample_rate)
                )
                    
            # Save the model
            
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                #save model to local directory
                pipeline.save_pretrained(output_dir)
               
                # log wandb artifact
                model_artifact = wandb.Artifact(
                    f'{wandb.run.id}-{args.project_name}',
                    type='model',
                    description='sonic-diffusion-model-256'
                    )
            
                model_artifact.add_dir(args.output_dir)
                wandb.log_artifact(
                    model_artifact,
                    aliases=[f'step_{global_step}', f'epoch_{epoch}']
                )
                if epoch == args.num_epochs - 1:
                    wandb.log({'Generated-Mel-Images-Table': wandb_table_image})
                    wandb.log({'Generated-Audio_Table': wandb_table_audio})
            
        accelerator.wait_for_everyone()

    accelerator.end_training()
    
#________________ END TRAINING LOOP __________________________________

# END MAIN FUNCTION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--run_name", type=str, default="default-run")
    parser.add_argument("--project_name", type=str, default="sonic-diffusion")
    parser.add_argument("--dataset_name", type=str, default="data")
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--num_images_in_table", type=int, default=6)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--restore_model", type=str, default=True, help="Should model continue training where it ended last run")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/model")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=2)
    
    parser.add_argument("--save_images_epochs", type=int, default=1, help ="Number of sample images to display")
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddim",
                        help="ddpm or ddim")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify a train data directory."
        )

    main(args)
