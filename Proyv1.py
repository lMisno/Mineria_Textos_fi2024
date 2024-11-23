import os
import json
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

directorio = os.path.dirname(os.path.abspath(__file__)) #directorio

CONFIG = {
    "max_lines": 50000,        # Lineas maximas a procesar (None para procesar todo el archivo)
    "min_rating": 3.0,         # Calificación minima para filtrar
    "batch_size": 10000,       # Tamaño de lotes para procesamiento
    "max_features": 10000,     # Maximo de palabras únicas para TF-IDF
    "num_topics": 10           # Numero de tópicos a generar
}

files_to_process = [ # archivos
    {"input": os.path.join(directorio, "CDs_and_Vinyl.jsonl"), 
     "jsonl_output": os.path.join(directorio, "topics_CD.jsonl")},
    {"input": os.path.join(directorio, "Toys_and_Games.jsonl"), 
     "jsonl_output": os.path.join(directorio, "topics_Toys.jsonl")},
]

stopwords = set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "you", "i", "we", "they", "this", "these", "those", "not",
    "but", "or", "if", "so", "such", "than", "then", "there", "their", "our",
    "your", "what", "when", "which", "who", "why", "how", "all", "any", "can",
    "do", "does", "did", "just", "more", "most", "no", "now", "only", "other",
    "some", "very", "my", "hes", "me", "her", "up", "had", "im", "here", "because",
])

##########

def normalize_text(text): #normalizar
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def detect_fraud_simple(review_text): # detectar fraude
    words = review_text.split()
    if len(words) < 5:
        return True  # muy corto
    if any(words.count(word) > len(words) * 0.5 for word in set(words)):
        return True  # muchas repeticiones
    return False

def detect_fraud_metadata(review):
    if review.get("helpful_votes", 0) == 0 and not review.get("verified_purchase", False):
        return True  # no tiene votos utiles ni es verificado
    return False

def detect_fraud_similarity(texts, threshold=0.9): #muy similar, tfidf
    vectorizer = TfidfVectorizer(max_features=CONFIG["max_features"])
    X = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(X)
    similar_pairs = [
        (i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
        if similarity_matrix[i, j] > threshold
    ]
    return set(i for pair in similar_pairs for i in pair)  # indices sospechosos

def process_large_file(file_path): # Procesar archivo en lotes
    texts, metadata, fraud_labels = [], [], []
    with open(file_path, 'r') as fp:
        batch, line_count = [], 0
        for line in fp:
            if CONFIG["max_lines"] and line_count >= CONFIG["max_lines"]:
                break
            review = json.loads(line.strip())
            line_count += 1
            if review.get("rating") is not None and review["rating"] <= CONFIG["min_rating"]:
                combined_text = f"{review.get('title', '')} {review.get('text', '')}"
                normalized_text = normalize_text(combined_text)
                words = normalized_text.split()
                cleaned_words = [word for word in words if word not in stopwords]
                text = " ".join(cleaned_words)
                batch.append(text)
                texts.append(text)
                metadata.append(review)
                fraud_labels.append( #detectar fraude
                    detect_fraud_simple(text) or detect_fraud_metadata(review)
                )
            if len(batch) >= CONFIG["batch_size"]:
                batch = []
    similarity_fraud = detect_fraud_similarity(texts) # por similitud
    for idx in similarity_fraud:
        fraud_labels[idx] = True  # Marcar como sospechosos
    return texts, metadata, fraud_labels

def analyze_topics(texts): # topicos
    vectorizer = TfidfVectorizer(max_features=CONFIG["max_features"])
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=CONFIG["num_topics"], random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = {
        f"Topic {i+1}": [feature_names[j] for j in topic.argsort()[-10:][::-1]]
        for i, topic in enumerate(lda.components_)
    }
    topic_probabilities = lda.transform(X)
    topics_assigned = [
        [feature_names[j] for j in lda.components_[i.argmax()].argsort()[-10:][::-1]]
        for i in topic_probabilities
    ]
    return topics, topics_assigned

#guardar resultados
def process_and_analyze(file_name, jsonl_output):
    print(f"Procesando archivo: {file_name}")
    texts, metadata, fraud_labels = process_large_file(file_name)
    print("Análisis de tópicos en curso...")
    topics, topics_assigned = analyze_topics(texts)
    print("Tópicos detectados:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")

    #formato jsonl
    with open(jsonl_output, 'w', encoding='utf-8') as jsonlfile:
        for idx, (text, meta, label, topic_keywords) in enumerate(zip(texts, metadata, fraud_labels, topics_assigned), start=1):
            jsonlfile.write(json.dumps({
                "Text": text,
                "Fraud_Label": bool(label),  # Convertir a booleano
                "Topics": topic_keywords,   # Agregar palabras clave del tópico
                "Rating": meta.get("rating"),
                "Helpful_Vote": meta.get("helpful_votes", 0),
                "Verified_Purchase": meta.get("verified_purchase", False)
            }) + '\n')
    print(f"Resultados guardados en {jsonl_output}")

#### PRINCIPAL ####

for file_config in files_to_process:
    process_and_analyze(
        file_config["input"], 
        file_config["jsonl_output"]
    )
