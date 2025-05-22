# custom_interface.py for CommonAccent English Accent Classifier
# Downloaded from: https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english/blob/main/custom_interface.py
# This file is required by the SpeechBrain foreign_class interface.

import torch
from speechbrain.pretrained.interfaces import Pretrained

class CustomEncoderWav2vec2Classifier(Pretrained):
    MODULES_NEEDED = ["model", "mean_var_norm", "label_encoder"]
    HPARAMS_NEEDED = ["sample_rate"]

    def classify_file(self, path):
        signal, fs = self.load_audio(path)
        return self.classify_batch(signal, fs)

    def classify_batch(self, signal, fs):
        if fs != self.hparams.sample_rate:
            signal = self.resample(signal, fs, self.hparams.sample_rate)
        signal = self.modules.mean_var_norm(signal, torch.tensor([1]))
        embeddings = self.modules.model.encode_batch(signal)
        out_prob = self.modules.model.classify_batch(embeddings)
        score, index = torch.max(out_prob, dim=1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab
