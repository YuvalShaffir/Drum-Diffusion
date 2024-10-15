# 🥁 Drum Diffusion: A Journey in Fine-Tuning AudioLDM2
This repository highlights the groundwork laid prior to our custom inference solution and documents our attempts, lessons learned, and the steps that led to failures and insights for further progress.
Here we have combined many of our repositories and Jupiter notbooks. Our final inference solution is in the HugginFace spaces repository that is at the end of this page.

**PAY ATTENTION**: DUE TO PROBLEM WITH THE GITHUB, CLONING ALL THE REPOSITORIES TOGETHER DIDN'T WORK WELL, 
SO I DOWNLOADED THEM AND PLACED THEM HERE.

### 🔗 Links to the repositories (This way to can see many of the comments):
- https://github.com/michaelpiro/training
- https://github.com/michaelpiro/drums_diff_inference
- https://github.com/YuvalShaffir/AudioLDM-training-finetuning (* a fork that I did to the many bugs in the AudioLDM training code)
- https://github.com/YuvalShaffir/Final-Project
- https://github.cs.huji.ac.il/yuval-shaffir/Drum-Diffusion
- https://github.com/HUJI-SCE/drums_diffusion


### 👨‍🎓👨‍🎓 The Team
**Team members**
- Yuval Shaffir
- Michael Pirogovsky

**Mentor**
- Guy Yariv

**Supervisor**
- Yossi Adi

### 📖 Project Description
The Drums Diffusion Project explores the use of state-of-the-art generative Diffusion models to add drum elements to existing music tracks. By leveraging pre-trained models such as AudioLDM2, AutoencoderKL, and CLAP, the project aims to investigate whether current generative AI tools can effectively enhance audio in this specific task. 

### 🚩 Initial Approach: Fine-Tuning AudioLDM2 U-Net

Our original approach focused on fine-tuning **AudioLDM2**'s U-Net to learn the distribution of latent representations of audio samples from a music dataset. We aimed to reduce the influence of the text prompt on the model's output, ensuring the model learns the latent space effectively. Unfortunately, several attempts ended unsuccessfully due to a lack of experience and knowledge. Below, we outline our efforts during this process:

#### 1. Dataset Creation

- **Datasets**: We used large audio datasets such as FMA and AudioCaps.
- **Demucs Model**: We processed each audio sample with the Demucs model (Alexandre Défossez et al., 2019) to separate channels and remove the drums.
- **Data Preparation**: The processed audio samples (without drums) were prepared according to AudioLDM's dataset creation guidelines. Additionally, a portion of the dataset was uploaded and integrated with the **Hugging Face DataSets API** for easier loading.

#### 2. AudioLDM2 Original Script Training

Our first idea was to modify the loss function and training objectives of the original AudioLDM2 script to focus on learning the latent representation of drum-less samples. Using the open-source code provided by **AudioLDM (Liu et al., 2023)**, we encountered and resolved several compatibility issues. However, even with access to high-end **Google GPUs**, the training process was too slow to deliver meaningful results.

#### 3. Modified AudioLDM2 U-Net Training

We explored multiple variations of U-Net training:
- **2D Conditional U-Net**: We trained the U-Net using a constant text prompt like "add drums" while feeding the model drumless audio.
- **1D Conditional U-Net**: We attempted to use 1D latent space, where random noise and latent representations were input.
- **Unconditional U-Net**: We tried training the U-Net with only random noise as input, excluding any text prompts.

None of these variations produced satisfactory results in generating drum patterns reliably.

#### 4. Training with Editing (Input Concatenation) Technique

Inspired by the **InstructPix2Pix** paper (Tim Brooks et al., 2022), we explored the idea of concatenating two latent vectors—one representing the condition (e.g., "add drums") and the other representing the expected result. This method, which allows for quick edits during the diffusion process, was implemented in AudioLDM for drum generation. However, our attempts to train the model using this approach led to unsatisfactory results.

### 🛰 Technologies:
- **Languages:** Python
- **Frameworks:** HuggingFace Spaces, Gradio
- **Libraries:** PyTorch, Torch-Audio, Librosa, NumPy, Diffusers.
- **Tools:** Git

### 🎼 Working Demo
https://huggingface.co/spaces/YuvalShaffir/real_demo
