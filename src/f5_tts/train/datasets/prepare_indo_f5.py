import os
import torchaudio
from datasets.arrow_writer import ArrowWriter
import json
import shutil

name_project = "" 
ch_tokenizer = False

path_data = "/home/ubuntu/F5-TTS/data"

path_project = os.path.join(path_data, name_project)
path_project_wavs = os.path.join(path_project, "wavs")
file_metadata = os.path.join(path_project, "metadata.csv")
file_raw = os.path.join(path_project, "raw.arrow")
file_duration = os.path.join(path_project, "duration.json")
file_vocab = os.path.join(path_project, "vocab.txt")

# if not os.path.isfile(file_metadata):
#     return "The file was not found in " + file_metadata, ""


def get_correct_audio_path(
    audio_input,
    base_path="wavs",
    supported_formats=("wav", "mp3", "aac", "flac", "m4a", "alac", "ogg", "aiff", "wma", "amr"),
):
    file_audio = None

    # Helper function to check if file has a supported extension
    def has_supported_extension(file_name):
        return any(file_name.endswith(f".{ext}") for ext in supported_formats)

    # Case 1: If it's a full path with a valid extension, use it directly
    if os.path.isabs(audio_input) and has_supported_extension(audio_input):
        file_audio = audio_input

    # Case 2: If it has a supported extension but is not a full path
    elif has_supported_extension(audio_input) and not os.path.isabs(audio_input):
        file_audio = os.path.join(base_path, audio_input)
        print("2")

    # Case 3: If only the name is given (no extension and not a full path)
    elif not has_supported_extension(audio_input) and not os.path.isabs(audio_input):
        print("3")
        for ext in supported_formats:
            potential_file = os.path.join(base_path, f"{audio_input}.{ext}")
            if os.path.exists(potential_file):
                file_audio = potential_file
                break
        else:
            file_audio = os.path.join(base_path, f"{audio_input}.{supported_formats[0]}")
    return file_audio

def get_audio_duration(audio_path):
    """Calculate the duration mono of an audio file."""
    audio, sample_rate = torchaudio.load(audio_path)
    return audio.shape[1] / sample_rate

def clear_text(text):
    """Clean and prepare text by lowering the case and stripping whitespace."""
    return text.lower().strip()

def format_seconds_to_hms(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds))

with open(file_metadata, "r", encoding="utf-8-sig") as f:
    data = f.read()

audio_path_list = []
text_list = []
duration_list = []

count = data.split("\n")
lenght = 0
result = []
error_files = []
text_vocab_set = set()
for line in data.split("\n"):
    sp_line = line.split("|")
    if len(sp_line) != 2:
        continue
    name_audio, text = sp_line[:2]

    file_audio = get_correct_audio_path(name_audio, path_project_wavs)

    if not os.path.isfile(file_audio):
        error_files.append([file_audio, "error path"])
        continue

    try:
        duration = get_audio_duration(file_audio)
    except Exception as e:
        error_files.append([file_audio, "duration"])
        print(f"Error processing {file_audio}: {e}")
        continue

    if duration < 1 or duration > 25:
        error_files.append([file_audio, "duration < 1 or > 25 "])
        continue
    if len(text) < 4:
        error_files.append([file_audio, "very small text len 3"])
        continue

    text = clear_text(text)
    # text = convert_char_to_pinyin([text], polyphone=True)[0] # RECHECK IF SKIPPABLE

    audio_path_list.append(file_audio)
    duration_list.append(duration)
    text_list.append(text)

    result.append({"audio_path": file_audio, "text": text, "duration": duration})
    if ch_tokenizer:
        text_vocab_set.update(list(text))

    lenght += duration

# if duration_list == []:
#     return f"Error: No audio files found in the specified path : {path_project_wavs}", ""

min_second = round(min(duration_list), 2)
max_second = round(max(duration_list), 2)

with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
    for line in result:
        writer.write(line)

with open(file_duration, "w") as f:
    json.dump({"duration": duration_list}, f, ensure_ascii=False)

new_vocal = ""
if not ch_tokenizer:
    if not os.path.isfile(file_vocab):
        file_vocab_finetune = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt")
        if not os.path.isfile(file_vocab_finetune):
            print("Error: Vocabulary file 'Emilia_ZH_EN_pinyin' not found!", "") 
        shutil.copy2(file_vocab_finetune, file_vocab)

    with open(file_vocab, "r", encoding="utf-8-sig") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

else:
    with open(file_vocab, "w", encoding="utf-8-sig") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")
            new_vocal += vocab + "\n"
    vocab_size = len(text_vocab_set)

if error_files != []:
    error_text = "\n".join([" = ".join(item) for item in error_files])
else:
    error_text = ""

print(
    f"prepare complete \nsamples : {len(text_list)}\ntime data : {format_seconds_to_hms(lenght)}\nmin sec : {min_second}\nmax sec : {max_second}\nfile_arrow : {file_raw}\nvocab : {vocab_size}\n{error_text}",
    new_vocal,
)