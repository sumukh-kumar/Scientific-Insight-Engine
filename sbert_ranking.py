from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_k_titles(main_title, other_titles, k=20):
    main_embedding = model.encode(main_title, convert_to_tensor=True)
    other_embeddings = model.encode(other_titles, convert_to_tensor=True)
    similarities = util.cos_sim(main_embedding, other_embeddings)[0]
    top_k = sorted(zip(other_titles, similarities), key=lambda x: x[1], reverse=True)[:k]
    return top_k

def get_top_k_full_matches(main_title, main_abstract, other_title_abstract_pairs, k=10):
    main_combined = main_title + " " + main_abstract
    other_combined = [title + " " + abstract for title, abstract in other_title_abstract_pairs]
    main_embedding = model.encode(main_combined, convert_to_tensor=True)
    other_embeddings = model.encode(other_combined, convert_to_tensor=True)
    similarities = util.cos_sim(main_embedding, other_embeddings)[0]
    top_k = sorted(zip(other_title_abstract_pairs, similarities), key=lambda x: x[1], reverse=True)[:k]
    return top_k

