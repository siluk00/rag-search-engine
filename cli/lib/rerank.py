import os, json, time

def individual_rerank(query, doc):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                                f"""Rate how well this movie matches the search query.

                                Query: "{query}"
                                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                                Consider:
                                - Direct relevance to query
                                - User intent (what they're looking for)
                                - Content appropriateness

                                Rate 0-10 (10 = perfect match).
                                Output ONLY the number in your response, no other text or explanation.

                                Score:"""
                                                      )         
    time.sleep(1)
    return response.text

def batch_rerank(query, doc_list_str):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                                f"""Rank the movies listed below by relevance to the following search query.

                                Query: "{query}"

                                Movies:
                                {doc_list_str}

                                Consider:
                                - Direct relevance to query
                                - User intent (what they're looking for)
                                - Content appropriateness

                                Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

                                For example:
                                [75, 12, 34, 2, 1]

                                Ranking:""")
    response = json.loads(response.text)
    return response

def cross_encoder_rerank(query, doc):
    from sentence_transformers import CrossEncoder
    pairs = []
    for d in doc:
        pairs.append([query, f"{d.get('title', '')} - {d.get('document', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    for i, d in enumerate(doc):
        d['new_score'] = scores[i]

    doc.sort(key=lambda x: x['new_score'], reverse=True)
    return doc

