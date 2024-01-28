import os
import shutil
import soundfile as sf
# import demucs_utils as dem
import demucs.separate
import librosa
import os
import eyed3
import numpy
import torch
import torchaudio
import matplotlib.pyplot as plt

FILE_TYPE = "--mp3"
DTYPE = '--float32'
TWO_STEMS = "--two-stems"
ROLE = "drums"
FLAG = "-o"
SAVE_PATH = "D:\Python Projects\Drum-Diffusion\dataset"
PAIRS = "pairs"
MODEL_FLAG = "-n"
MODEL = "mdx_extra"
NEW_DIR_NAME = "mdx_extra"
DEMUCS_OUT_DIR = os.path.join(SAVE_PATH, NEW_DIR_NAME)
PAIRS_DIR = os.path.join(SAVE_PATH, PAIRS)
LEN_IN_SEC = 5
OVERLAP_IN_SEC = 0.25
DUMP_SHORTER = True
EXT = '.mp3'
NO_DRUMS_EXT = 'no_drums' + EXT
DRUMS_EXT = 'drums' + EXT


def extract_files_to_directory(in_path, out_path):
    """
    Extracts all the files from in_path and its subdirectories to out_path.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for root, _, files in os.walk(in_path):
        for file in files:
            # Full path of the file in the source directory
            src_file = os.path.join(root, file)
            # Full path of the file in the destination directory
            dest_file = os.path.join(out_path, file)

            # Check if a file with the same name already exists in the destination
            if os.path.exists(dest_file):
                # Modify the filename to prevent overwriting
                base, ext = os.path.splitext(file)
                counter = 1
                new_dest_file = os.path.join(out_path, f"{base}_{counter}{ext}")

                # Increment counter until a new, unused name is found
                while os.path.exists(new_dest_file):
                    counter += 1
                    new_dest_file = os.path.join(out_path, f"{base}_{counter}{ext}")

                dest_file = new_dest_file

            shutil.copy2(src_file, dest_file)

# Example usage:
# extract_files_to_directory('path/to/input/directory', 'path/to/output/directory')

# Note: Replace 'path/to/input/directory' and 'path/to/output/directory' with the actual paths.


def cut_audio_files_librosa(in_path, length_in_sec, overlap_in_sec, out_path):
    """
    Cuts each audio file in in_path into tracks of specified length with overlap,
    and saves them to out_path using librosa.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']

    for file in os.listdir(in_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_formats:
            audio_path = os.path.join(in_path, file)
            audio, sr = librosa.load(audio_path, sr=None)  # Load audio with its native sampling rate

            length_samples = int(length_in_sec * sr)  # Convert to number of samples
            overlap_samples = int(overlap_in_sec * sr)  # Convert to number of samples

            start = 0
            part = 1
            while start + length_samples <= len(audio):
                # Calculate end position considering the overlap
                end = start + length_samples
                if end > len(audio):
                    end = len(audio)

                # Extract part of the audio
                chunk = audio[start:end]

                # Save the extracted part
                chunk_name = f"{os.path.splitext(file)[0]}_part{part}{ext}"
                chunk_path = os.path.join(out_path, chunk_name)
                sf.write(chunk_path, chunk, int(sr))
                part += 1
                start = end - overlap_samples  # Move start for the next chunk considering the overlap

# Example usage:
# cut_audio_files_librosa('path/to/input/directory', 30, 5, 'path/to/output/directory')

# Note: Replace the paths and time values with actual values as needed.


# def extract_and_cut_audio_files(in_path, length_in_sec, overlap_in_sec, out_path,file1,file2):
#     """
#     Extracts all the audio files from in_path and its subdirectories, then cuts each audio file into
#     segments of specified length with overlap, and saves them to out_path using librosa.
#     """
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#
#     supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
#
#     for root, _, files in os.walk(in_path):
#         for file in files:
#             ext = os.path.splitext(file)[1].lower()
#             if ext in supported_formats:
#                 audio_path = os.path.join(root, file)
#                 audio, sr = librosa.load(audio_path, sr=None)  # Load audio with its native sampling rate
#
#                 length_samples = int(length_in_sec * sr)  # Convert to number of samples
#                 overlap_samples = int(overlap_in_sec * sr)  # Convert to number of samples
#
#                 start = 0
#                 part = 1
#                 while start + length_samples <= len(audio):
#                     # Calculate end position considering the overlap
#                     end = start + length_samples
#                     if end > len(audio):
#                         end = len(audio)
#
#                     # Extract part of the audio
#                     chunk = audio[start:end]
#
#                     # Create a directory for each original file to store its segments
#                     file_dir = os.path.splitext(file)[0]
#                     # segment_dir = os.path.join(out_path, file_dir)
#                     # CHANGE IT - SO IT WILL STORE EVERY FILE IN THE SAME DIR
#                     segment_dir = os.path.join(out_path, '')
#                     if not os.path.exists(segment_dir):
#                         os.makedirs(segment_dir)
#
#                     # Save the extracted part
#                     chunk_name = f"{file_dir}_part{part}{ext}"
#                     chunk_path = os.path.join(segment_dir, chunk_name)
#                     sf.write(chunk_path, chunk, sr)
#
#                     part += 1
#                     start = end - overlap_samples  # Move start for the next chunk considering the overlap

# Example usage:
# extract_and_cut_audio_files('path/to/input/directory', 30, 5, 'path/to/output/directory')

# Note: Replace the paths and time values with actual values as needed.




SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
# UNSUPPORTED_GENRES = ['Noise', 'Speech', 'Satire', '']
SAMPLE_RATE = 44100



def extract_and_cut_audio_files2(in_path, length_in_sec, overlap_in_sec, out_path, dump_shorter=True, sample_rate = SAMPLE_RATE):
    """
    Extracts all the audio files from in_path and its subdirectories, then cuts each audio file into
    segments of specified length with overlap, and saves them to out_path using librosa. If dump_shorter
    is True, segments shorter than length_in_sec are not saved.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
    for root, _, files in os.walk(in_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                audio_path = os.path.join(root, file)
                audio, sr = librosa.load(audio_path, sr=None)  # Load audio with its native sampling rate
                if int(sr) != sample_rate:
                    break
                if not sample_rate:
                    length_samples = int(length_in_sec * sr)  # Convert to number of samples
                    overlap_samples = int(overlap_in_sec * sr)
                else:
                    length_samples = int(length_in_sec * sample_rate)  # Convert to number of samples
                    overlap_samples = int(overlap_in_sec * sample_rate)

                start = 0
                part = 1
                while start < len(audio):
                    end = start + length_samples
                    if end > len(audio):
                        end = len(audio)

                    # If dump_shorter is True, don't save the last shorter segment
                    if dump_shorter and end - start < length_samples:
                        break

                    # Extract part of the audio
                    chunk = audio[start:end]

                    # Create a directory for each original file to store its segments
                    file_dir = os.path.splitext(file)[0]
                    segment_dir = os.path.join(out_path, file_dir)
                    if not os.path.exists(segment_dir):
                        os.makedirs(segment_dir)

                    # Save the extracted part
                    chunk_name = f"{file_dir}_part{part}{ext}"
                    chunk_path = os.path.join(segment_dir, chunk_name)
                    # sf.write(chunk_path, chunk, sr)
                    sf.write(chunk_path, chunk, int(sr))
                    demucs.separate.main(
                        [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL,
                        chunk_path])
                    os.rename(chunk_path,os.path.join(os.path.join(SAVE_PATH,NEW_DIR_NAME),chunk_name))
                    part += 1
                    start = end - overlap_samples  # Move start for the next chunk considering the overlap



def cut_save_audio():
    for dirs in os.listdir(DEMUCS_OUT_DIR):
        dir_path = os.path.join(DEMUCS_OUT_DIR,dirs)
        if os.path.isdir(dir_path):
            files =  os.listdir(dir_path)
            if len(files) != 2:
                continue
            file1 = os.path.join(dir_path,files[0])
            file2 = os.path.join(dir_path,files[1])
            if check_file(file1) and check_file(file2):
                cut_song(file1,file2,dirs)


def cut_song(file1,file2,dir_name):
    # print(f'f1 name : {file1}\n')
    # print(f'f2 name : {file2}\n')
    audio1, sr1 = torchaudio.load(file1) #no drums
    audio2, sr2 = torchaudio.load(file2) #with drums
    length_samples = int(LEN_IN_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_IN_SEC * SAMPLE_RATE)
    # print(audio1.shape)
    # print(audio2.shape)
    start = 0
    part = 1
    if len(audio1 == 2):
        length = len(audio1[0])
    else:
        length = len(audio1)
    while start < length:
        end = start + length_samples
        if end > length:
            end = length

        # If dump_shorter is True, don't save the last shorter segment
        if DUMP_SHORTER and end - start < length_samples:
            break

        # Extract part of the audio
        chunk1 = audio1[:,start:end]
        # print(f'chank1 shape: {chunk1.shape}')
        chunk2 = audio2[:,start:end]

        # Create a directory for each original file to store its segments
        file_dir = dir_name
        chunk_name = f"{file_dir}_part{part}"
        segment_dir = os.path.join(PAIRS_DIR, chunk_name)
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)

        # Save the extracted part
        # chunk_path = os.path.join(segment_dir, chunk_name)
        # sf.write(chunk_path, chunk, sr)
        print(f'segment dir : {segment_dir}\n')
        # print(f'first file without drums : {segment_dir + EXT}\n')
        # print(f'second file with drums : {segment_dir + EXT}\n')
        f1_save_path = os.path.join(segment_dir,NO_DRUMS_EXT)
        f2_save_path = os.path.join(segment_dir,DRUMS_EXT)
        print(f'first file without drums : {f1_save_path}\n')
        print(f'second file with drums : {f1_save_path}\n')
        torchaudio.save(f1_save_path, chunk1, SAMPLE_RATE)
        torchaudio.save(f2_save_path, chunk2, SAMPLE_RATE)
        # sf.write(segment_dir + EXT, chunk1, int(sr1))
        # sf.write(segment_dir + EXT, chunk2, int(sr2))

        part += 1
        start = end - overlap_samples  # Move start for the next chunk considering the overlap



#todo: solve the bug with demucs files, change the names of the saved files, ##DONE##!

def check_file(file_path):
    try:
        audio, sr = torchaudio.load(file_path)  # Load audio with its native sampling rate
        # print(len(audio))
        if sr != SAMPLE_RATE:
            return False
        return True
    except:
        print(f'{file_path} is corrupted')
        return False


def demucs_that_fatBass_file(file_path):
    demucs.separate.main(
        [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL,
        file_path])

def apply_demucs_on_dir(in_path, out_path, sample_rate = SAMPLE_RATE):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
    # supported_genres = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
    for root, _, files in os.walk(in_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FORMATS:
                audio_path = os.path.join(root, file)
                if check_file(audio_path):
                    demucs_that_fatBass_file(audio_path)






# Example usage:
# extract_and_cut_audio_files('path/to/input/directory', 30, 5, 'path/to/output/directory', dump_shorter=True)

# Note: Replace the paths and time values with actual values as needed.

def check_sample_rate(in_path):
    rates_dict = dict()
    rates_set = set()
    bad_files = []
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
    for root, _, files in os.walk(in_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                audio_path = os.path.join(root, file)
                try:
                    audio, sr = librosa.load(audio_path, sr=None)  # Load audio with its native sampling rate
                except:
                    bad_files.append(audio_path)
                    continue
                if sr in rates_set:
                    val = rates_dict[sr] + 1
                    rates_dict[sr] = val
                else:
                    rates_dict[sr] = 1
                    rates_set.add(sr)
    with open('rates.txt', 'w') as f:
        for rate in rates_dict.keys():
            f.write(f'sample rate: {rate} , number of tracks: {rates_dict[rate]}\n')
    with open('bad_files.txt', 'w') as f:
        for file in bad_files:
            f.write(f'{file}\n')


def check_genres(in_path):
    genres_dict = dict()
    genres_set = set()
    bad_files = []
    counter = 0
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
    for root, _, files in os.walk(in_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                audio_path = os.path.join(root, file)
                try:
                    audiofile = eyed3.load(audio_path)
                except:
                    bad_files.append(audio_path)
                    continue
                if audiofile.tag.genre:
                    if audiofile.tag.genre.name in genres_set:
                        val = genres_dict[audiofile.tag.genre.name] + 1
                        genres_dict[audiofile.tag.genre.name] = val
                    else:
                        genres_dict[audiofile.tag.genre.name] = 1
                        genres_set.add(audiofile.tag.genre.name)
                else:
                    counter+= 1
    with open('/Users/mac/pythonProject1/pythonProject/utils/genres.txt', 'w') as f:
        for genre in genres_dict.keys():
            f.write(f'genre: {genre} , number of tracks: {genres_dict[genre]}\n')
        f.write(f'genre: DONT HAVE ONE , number of tracks: {counter}\n')
    with open('/Users/mac/pythonProject1/pythonProject/utils/bad_files.txt', 'w') as f:
        for file in bad_files:
            f.write(f'{file}\n')



def check_one_info_in_fma_dir(func,in_path,file1,file2):
    def inner1(*args, **kwargs):
        info_dict = dict()
        info_set = set()
        issues = []
        supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
        for root, _, files in os.walk(in_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_formats:
                    audio_path = os.path.join(root, file)
                    try:
                        audio, sr = librosa.load(audio_path, sr=None , mono=True, offset=0.0, duration=None, dtype= 'float32', res_type='soxr_hq')  # Load audio with its native sampling rate
                    except:
                        issues.append(audio_path)
                        continue
                    x = func(audio,sr)
                    if x in info_set:
                        val = info_dict[x] + 1
                        info_dict[x] = val
                    else:
                        info_dict[x] = 1
                        info_set.add(x)
        with open(file1, 'w') as f:
            for info in info_dict.keys():
                f.write(f'info: {info} , number of tracks: {info_dict[info]}\n')
        with open(file2, 'w') as f:
            for issue in issues:
                f.write(f'problem in: {issue}\n')
    return inner1


def apply_func_on_every_file(func):
    def inner(in_path,*args, **kwargs):
        issues = []
        supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac']
        for root, _, files in os.walk(in_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_formats:
                    audio_path = os.path.join(root, file)
                    try:
                        func(audio_path,*args, **kwargs)
                    except:
                        issues.append(audio_path)
                        continue
    return inner

@apply_func_on_every_file
def check_dtype(audio_path,*args, **kwargs):
    audio, sr = librosa.load(audio_path, sr=None, mono=True, offset=0.0, duration=None, dtype='float32',
                             res_type='soxr_hq')  # Load audio with its native sampling rate
    print (audio.dtype)

def plot_spectrogram(specgram,num ,title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.figure(num)

def melspec():
    n_fft = 512
    win_length = None
    hop_length = 256
    n_mels = 512
    sample_rate = SAMPLE_RATE

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    return mel_spectrogram


def create_save_mel_spec():
    waveform1, sample_rate = torchaudio.load("/Users/mac/pythonProject1/pythonProject/utils/pairs/136466_part3/drums.mp3", normalize=True)
    waveform2, sample_rate = torchaudio.load("/Users/mac/pythonProject1/pythonProject/utils/pairs/136466_part3/no_drums.mp3", normalize=True)
    transform = melspec()
    mono1 = torch.mean(waveform1,dim=0,keepdim=False)
    mono2 = torch.mean(waveform2,dim=0,keepdim=False)
    mel1 = transform(waveform1)  # (channel, n_mels, time)
    # mel2 = transform(waveform2)  # (channel, n_mels, time)
    mel3 = transform(mono1)  # (channel, n_mels, time)
    mel4 = transform(mono2)  # (channel, n_mels, time)
    mel5 = transform(mono1+mono2)
    # print(mel2.shape)
    # plot_spectrogram(mel1[0],1, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    # plot_spectrogram(mel1[1],2, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    # plot_spectrogram(mel2[1],3, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    # plot_spectrogram(mel2[1],4, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plot_spectrogram(mel3,1, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plot_spectrogram(mel4,2, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plot_spectrogram(mel5,3, title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plt.show()




if __name__ == '__main__':
    # audio,sr = torchaudio.load('/Users/mac/pythonProject1/pythonProject/utils/mdx_extra/136466/drums.mp3')
    # print(audio)
    # print(sr)

    # cut_save_audio()
    # create_save_mel_spec()

    # demucs.separate.main(
    #     [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL,
    #     "/Users/mac/pythonProject1/pythonProject/utils/demo_set/136/136054.mp3", "/Users/mac/pythonProject1/pythonProject/utils/demo_set/137/137151.mp3"])
    # extract_and_cut_audio_files2('/Users/mac/pythonProject1/pythonProject/utils/small_demo_set',10,0,os.path.join(SAVE_PATH,NEW_DIR_NAME))
    # check_sample_rate('/Users/mac/pythonProject1/pythonProject/utils/demo_set')
    # check_sample_rate('/Users/mac/Downloads/fma_small')
    # method = check_one_info_in_fma_dir(check_dtype,'/Users/mac/pythonProject1/pythonProject/utils/wow','dtype.txt','issues_dtype.txt')
    # method()
    # check_dtype('/Users/mac/pythonProject1/pythonProject/utils/wow')
    # check_genres('/Users/mac/Downloads/fma_small')
    apply_demucs_on_dir('D:/fma_small/fma_small' , SAVE_PATH)

# Example usage:
# copy_directory_contents('path/to/input/directory', 'path/to/output/directory')

# Note: Replace 'path/to/input/directory' and 'path/to/output/directory' with the actual paths.
