from openai import OpenAI
import os
from dotenv import load_dotenv
import string

load_dotenv()
key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=key)


# clean text
def clean_text(text):
    """
    Clean the text by removing punctuation.
    """
    if isinstance(text, str):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text
    else:
        return text
    
def get_embedding(input: str) -> str|None:
    """
    Get the embedding for a given text string using OpenAI's API.
    """
    print(input)
    if isinstance(input, str):
        input = clean_text(input)
        response = client.embeddings.create(
            input=input,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    else:
        return None