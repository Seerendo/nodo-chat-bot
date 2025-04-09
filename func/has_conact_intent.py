import spacy
import os

model_path = os.path.join("spacy_models", "es_core_news_sm", "es_core_news_sm-3.8.0")
nlp = spacy.load(model_path)

INTENT_LEMMAS = {"contactar", "enviar", "hablar", "escribir", "comunicar", "mandar"}

def has_contact_intent(text: str) -> bool:
    doc = nlp(text.lower())
    for token in doc:
        if token.lemma_ in INTENT_LEMMAS:
            return True
    return False


if __name__ == "__main__":
    print(has_contact_intent("Contactó el cliente"))