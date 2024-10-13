import os
import csv


def create_dataset_csv(path_to_orig,path_to_sep, csv_file_name='dataset.csv'):
    # Define a list of audio file extensions
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']

    # # Ensure the directory exists
    # if not os.path.exists(dir_path):
    #     raise ValueError("The provided directory does not exist")
    file_names = []
    dirs = os.listdir(path_to_orig)
    for dir in dirs:
        path = os.path.join(path_to_orig,dir)
        files_in_dir = os.listdir(path)
        path_to_files = [os.path.join(path,file_name) for file_name in files_in_dir]
        file_names += path_to_files

    with open(csv_file_name, 'w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['file_name','drums_path','no_drums_path'])
        print("collecting files...")

        for i in range(len(file_names)):
            ext = os.path.splitext(file_names[i])[1].lower()
            name = os.path.basename(os.path.splitext(file_names[i])[0].lower())
            separated_dir = os.path.join(path_to_sep,name)
            if not os.path.exists(separated_dir):
                print(f"The directory {separated_dir} does not exist, file didn't separated")
            else:
                d_path = os.path.join(separated_dir,f"drums{ext}")
                no_d_path = os.path.join(separated_dir,f"no_drums{ext}")
                is_drums = os.path.exists(d_path)
                is_no_drums = os.path.exists(no_d_path)
                if is_drums and is_no_drums:
                    writer.writerow([file_names[i], d_path, no_d_path])
                else:
                    print(f"The directory {separated_dir} exist!, but file didn't separated")


# Usage Example
# create_dataset_csv('/path/to/your/directory')
# import diffusers.pipelines.audioldm2.modeling_audioldm2 as A
# import torch
# def create_unet():
#     ynet = A.AudioLDM2UNet2DConditionModel(
#         sample_size=256,
#         in_channels=8,
#         out_channels=8,
#         flip_sin_to_cos=True,
#         freq_shift=0,
#         down_block_types=(
#             "DownBlock2D",
#             "CrossAttnDownBlock2D",
#             "CrossAttnDownBlock2D",
#             "CrossAttnDownBlock2D",
#         ),
#         mid_block_type="UNetMidBlock2DCrossAttn",
#         up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
#         only_cross_attention=False,
#         block_out_channels=(128, 256, 512, 1024),
#         layers_per_block=2,
#         downsample_padding=1,
#         mid_block_scale_factor=1,
#         act_fn="silu",
#         norm_num_groups=32,
#         norm_eps=1e-5,
#         # cross_attention_dim = ([None,1024,64],[None,1024,64],[None,1024,64],[None,1024,64]),
#         cross_attention_dim=[[None, None], [None, None], [None, None], [None, None]],
#         # cross_attention_dim = [[None,64],[None,64],[None,64],[None,64]],
#         transformer_layers_per_block=1,
#         attention_head_dim=8,
#         num_attention_heads=None,
#         use_linear_projection=False,
#         class_embed_type=None,
#         num_class_embeds=None,
#         upcast_attention=False,
#         resnet_time_scale_shift="default",
#         time_embedding_type="positional",
#         time_embedding_dim=None,
#         time_embedding_act_fn=None,
#         timestep_post_act=None,
#         time_cond_proj_dim=None,
#         conv_in_kernel=3,
#         conv_out_kernel=3,
#         projection_class_embeddings_input_dim=None,
#         class_embeddings_concat=False,
#     )
#     torch.save(ynet.half(),'/Users/mac/pythonProject1/pythonProject/train_file/models/try_unet')

if __name__ == '__main__':
    path_to_orig = "D:\\yuval.shaffir\\fma_small"
    path_to_sep = "D:\\yuval.shaffir\\separated\\mdx_extra"
    csv_file = "D:\\yuval.shaffir\\annotation_check.csv"

    create_dataset_csv(path_to_orig,path_to_sep,csv_file)
    # with open("anno.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     # Write the header
    #     writer.writerow(['file_name', 'path_from_dir', 'drums_path', 'no_drums_path'])
    #     a = '136054.mp3,/Users/mac/pythonProject1/pythonProject/utils/small_demo_set/1/136054.mp3,/Users/mac/pythonProject1/pythonProject/utils/mdx_extra/136054/drums.mp3,/Users/mac/pythonProject1/pythonProject/utils/mdx_extra/136054/no_drums.mp3'
    #     b = "136466.mp3,/Users/mac/pythonProject1/pythonProject/utils/small_demo_set/2/136466.mp3,/Users/mac/pythonProject1/pythonProject/utils/mdx_extra/136466/drums.mp3,/Users/mac/pythonProject1/pythonProject/utils/mdx_extra/136466/no_drums.mp3"
    #     writer.writerow(a.split(','))
    #     writer.writerow(b.split(','))