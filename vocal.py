import gradio as gr
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import soundfile as sf

# Charger modèle et processeur
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)

# Fonction qui prend un audio et génère une réponse
def analyse_audio(audio):
    # Lire l'audio capturé
    audio_array, sample_rate = sf.read(audio)

    # Créer un prompt simple (on pourrait le rendre dynamique si tu veux)
    text = "<|audio_bos|><|AUDIO|><|audio_eos|>Please describe what you hear."

    # Préparer l'entrée pour le modèle
    inputs = processor(text=text, audios=audio_array, return_tensors="pt", padding=True)

    # Envoyer sur le bon device (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Génération
    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    # Décodage
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

# Interface Gradio
interface = gr.Interface(
    fn=analyse_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="🎙️ Analyse vocale avec Qwen2-Audio",
    description="Clique sur 'Record', parle, et laisse l'IA analyser ton audio ! 🚀",
)

# Lancer
interface.launch()
