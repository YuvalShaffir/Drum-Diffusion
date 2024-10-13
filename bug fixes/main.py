import torch
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid

from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

@dataclass
class TrainingConfig:
    # AUDIO CONFIGS
    n_fft = 1024
    hop_length = 128
    win_length = 1024
    n_mels = 64
    target_sr = 22050
    fmax = int(target_sr/ 2)
    fmin = 0

    # FIRST OPTION - TO DEFINE THE WANTED LENGTH OF THE SPEC WITH target_length = 1024
    target_length = 1024
    num_samples = (target_length - 1) * hop_length # BECAUSE OF THE WAY FFT IN LIBROSA WORKS
    # IF THE CENTER ARG IN THE MELSPEC FUNC IS FALSE THEN:
        # num_samples = (target_length - 1) * hop_length + win_length
    length_in_sec = num_samples / target_sr

    #OTHER OPTION - TO DEFINE THE DURATION OF THE AUDIO, BUT THE SPECTOGRAM MIGHT HAVE UGLY LENGTH
    audio_len_sec = 5
    num_samples = int(audio_len_sec * target_sr)

    loss_noise_factor = 0.3
    loss_diff_drums_factor = 1 - loss_noise_factor
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 2000
    eval_epoch = 10
    save_model_epochs = 5
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "/drums_diff/out_dir"  # the model name locally and on the HF Hub
    models_dir = "models_dir"
    dataset_dir = "dataset_dir"
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0




def load_pretrained_models(dir_path):
    """
    Loads pretrained UNet, AutoencoderKL, and Vocoder models from the specified directory path.

    Args:
    - dir_path (str): Path to the directory containing the pretrained models.

    Returns:
    - models (dict): A dictionary containing the loaded models.
    """
    models = {
        "unet": torch.load(f"{dir_path}/unet.pt"),
        "autoencoder_kl": torch.load(f"{dir_path}/autoencoder_kl.pt"),
        "vocoder": torch.load(f"{dir_path}/vocoder.pt"),
        "scheduler": torch.load("scheduler.pt")
    }
    return models



def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, unet, vae, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    unet, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            #TODO: GET ITEM RETURN (NO_DRUM_SPEC,DRUM_SPEC)
            batch_no_drum_spec, batch_drum_spec = batch
            difference_spec = batch_no_drum_spec - batch_drum_spec

            latents_drums = vae.encode(batch_drum_spec).latent_dist.sample() * 0.18215
            latents_no_drums = vae.encode(batch_no_drum_spec).latent_dist.sample() * 0.18215

            # Sample noise to add to the images
            noise = torch.randn_like(latents_no_drums.shape, device=batch_drum_spec.device)
            # noise_minus_drums = noise + latents_drums
            batch_size = batch_drum_spec.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=batch_drum_spec.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noisy_drums_latents = noise_scheduler.add_noise(latents_drums, noise, timesteps)
            # difference_noise_spec = noise_scheduler.add_noise(batch_drum_spec - batch_no_drum_spec,noise,timesteps)
            noisy_no_drums_latents = noise_scheduler.add_noise(latents_no_drums, noise, timesteps)
            with accelerator.accumulate(unet):
                # Predict the noise residual
                noise_pred = unet(noisy_no_drums_latents,timesteps, encoder_hidden_states=batch_no_drum_spec, return_dict=False)[0]

                loss = config.loss_noise_factor*F.mse_loss(noise_pred, noise) + \
                    config.loss_diff_drums_factor*F.mse_loss(noise_pred,noisy_no_drums_latents)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(unet), scheduler=noise_scheduler)

            if (epoch + 1) % config.eval_epoch == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

if __name__ == '__main__':
    config = TrainingConfig()
    dataset = Dataset(config.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    models = load_pretrained_models(config.models_dir)
    unet = models["unet"]
    unet.train()
    noise_scheduler = models["scheduler"]
    vae = models["vae"]
    vae.requires_grad_(False)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataset) * config.num_epochs),
    )
    train_loop(config, unet, vae, noise_scheduler, optimizer, dataloader, lr_scheduler)



