from transformers import pipeline

# Indique à Hugging Face d'utiliser uniquement PyTorch
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="pt")

# Tester sur un exemple de texte
result = pipe("I love using Hugging Face models, they are awesome!")

# Afficher le résultat
print(result)
