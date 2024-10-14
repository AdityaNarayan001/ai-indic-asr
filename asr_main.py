import torch
import os
import nemo.collections.asr as nemo_asr

from utility.audio_to_mono import audio_to_mono

MODEL_PATH = "/Users/aditya.narayan/Desktop/speechToSpeech/indic-asr/asr_model/ai4b_indicConformer_kn.nemo"



RAW_AUDIO = "/Users/aditya.narayan/Desktop/speechToSpeech/indic-asr/raw_audio/sample_audio.wav"
PROCESSED_AUDIO_0 = "/Users/aditya.narayan/Desktop/speechToSpeech/indic-asr/processed_audio/mono_left.wav"
PROCESSED_AUDIO_1 = "/Users/aditya.narayan/Desktop/speechToSpeech/indic-asr/processed_audio/mono_right.wav"

if len(os.listdir("/Users/aditya.narayan/Desktop/speechToSpeech/indic-asr/processed_audio")) == 1:
    print("🔴 No Mono file found !")
    audio_to_mono(RAW_AUDIO, PROCESSED_AUDIO_0, PROCESSED_AUDIO_1)
    print("🟢 Mono file created.")
else:
    print("🟢 Mono files found.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=MODEL_PATH, strict=False)
model.freeze()
model = model.to(device)

model.cur_decoder = 'ctc'
ctc_text = model.transcribe([PROCESSED_AUDIO_0], batch_size=1,logprobs=False, language_id='kn')[0]
print('Text : ',ctc_text)


