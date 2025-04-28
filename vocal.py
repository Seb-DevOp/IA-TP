import torch

import librosa

import gradio as gr

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
 
# Charger modèle et processeur

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
 
# Fonction de traitement de l'audio

def analyse_audio(file_obj):

    if file_obj is None:

        return "Erreur : Aucun fichier audio reçu."
 
    # Charger l'audio

    audio_array, _ = librosa.load(file_obj.name, sr=processor.feature_extractor.sampling_rate)
 
    # Préparer la conversation

    conversation = [

        {"role": "user", "content": [

            {"type": "audio", "audio": audio_array},

        ]}

    ]
 
    # Générer le texte ChatML

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
 
    # Préparer les entrées

    inputs = processor(text=text, audio=audio_array, return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
 
    # Générer

    generate_ids = model.generate(**inputs, max_new_tokens=256)

    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
 
    # Décoder

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response
 
# Interface Gradio

interface = gr.Interface(

    fn=analyse_audio,

    inputs=gr.File(label="Choisissez votre fichier audio (WAV ou MP3)"),

    outputs="text",

    title="🎤 Analyse audio avec Qwen2-Audio",

    description="Téléverse un fichier audio (.wav ou .mp3), l'IA analyse ce qu'elle entend ! 🚀"

)
 
# Lancer

interface.launch()

 