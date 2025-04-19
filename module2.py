import arxiv
import fitz  # PyMuPDF
import requests
from io import BytesIO
import re
import nltk
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

# Load SciBERT model
model = SentenceTransformer('allenai-specter')

def get_arxiv_pdf_link(title):
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for result in search.results():
        return result.pdf_url  # Direct PDF link

    return None

def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_bytes):
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    with open("paperr.txt", 'w', encoding='utf-8') as fp:
        fp.write(text)  
    return text


def smart_split_paragraphs1(text):
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Cut off everything after "References" (case-insensitive)
    text = re.split(r'\n\s*references\s*\n', text, flags=re.IGNORECASE)[0]

    lines = text.split('\n')
    paragraphs = []
    current_paragraph = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue  # skip empty lines

        # If line looks like a heading or section title
        if re.match(r'^(abstract|introduction|[0-9]+\.\s)', stripped.lower()):
            if current_paragraph:
                combined = ' '.join(current_paragraph).strip()
                if len(combined.split()) > 2:
                    paragraphs.append(combined)
                current_paragraph = []
            paragraphs.append(stripped)
        elif current_paragraph and stripped[0].isupper() and current_paragraph[-1].endswith('.'):
            # New paragraph if line starts with a capital and previous line ended with a period
            combined = ' '.join(current_paragraph).strip()
            if len(combined.split()) > 2:
                paragraphs.append(combined)
            current_paragraph = [stripped]
        else:
            current_paragraph.append(stripped)

    # Final paragraph
    if current_paragraph:
        combined = ' '.join(current_paragraph).strip()
        if len(combined.split()) > 2:
            paragraphs.append(combined)

    return paragraphs


def get_relevant_paragraphs(text, keyword, top_k=5):

    paragraphs = smart_split_paragraphs1(text)
    
    # Skip if not enough paragraphs
    if len(paragraphs) == 0:
        return []

    actual_k = min(top_k, len(paragraphs))  # avoid crash

    
    paragraphs = [i for i in paragraphs if len(i.split()) > 10]


    print(len(paragraphs))


    # paragraphs=paragraphs[:196]

    
    para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(keyword_embedding, para_embeddings)[0]
    top_results = cos_scores.topk(k=actual_k)

    matches = [paragraphs[idx] for idx in top_results.indices]
    scores = [cos_scores[idx].item() for idx in top_results.indices]
    return list(zip(matches, scores))

def get_relevant_paragraphs1(text, keyword, threshold=0.8):
    paragraphs = smart_split_paragraphs1(text)
    
    # Filter out short paragraphs
    paragraphs = [p for p in paragraphs if len(p.split()) > 10]
    
    if not paragraphs:
        return []

    para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)

    # Calculate cosine similarity scores
    cos_scores = util.pytorch_cos_sim(keyword_embedding, para_embeddings)[0]

    # Filter by threshold
    matches = [(para, cos_scores[i].item()) for i, para in enumerate(paragraphs) if cos_scores[i] >= threshold]

    # Optionally sort matches by score descending
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches

def extract_info_semantically(paper_urls, keywords, top_k=5):
    results = []

    for url in paper_urls:
        paper_id = url.split("/")[-1].replace(".pdf", "")
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        pdf_url = paper.pdf_url
        print(f"📄 Processing: {paper.title}")

        pdf_bytes = download_pdf(pdf_url)
        full_text = extract_text_from_pdf(pdf_bytes)

        # print(full_text)

        paper_result = {"title": paper.title, "paper_id": paper_id, "keywords": {}}
        for keyword in keywords:
            # matches = get_relevant_paragraphs(full_text, keyword, top_k=top_k)
            matches = get_relevant_paragraphs1(full_text, keyword)
            paper_result["keywords"][keyword] = matches
        results.append(paper_result)

    return results


def para_extractor(paper_title,paper_urls,keywords):

    pdf_link = get_arxiv_pdf_link(paper_title)

    if pdf_link:
        print(f"PDF Link: {pdf_link}")
    else:
        print("Paper not found.")


    info = extract_info_semantically(paper_urls, keywords)

    for paper in info:
        print(f"\n📘 Paper: {paper['title']}")
        for keyword, matches in paper['keywords'].items():
            print(f"\n🔑 Keyword: {keyword}")
            for para, score in matches:
                print(f"\n→ Score: {score:.2f}\n{para}\n{'-'*80}")
    return info

if __name__== "__main__":
    paper_urls = ["https://arxiv.org/pdf/2103.00020.pdf"]
    keywords = ["Language","latent space", "visual", "transformer","CLIP"]
    
    info = extract_info_semantically(paper_urls, keywords, top_k=3)

    paper_id = paper_urls[0].split("/")[-1].replace(".pdf", "")
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    pdf_url = paper.pdf_url
    print(f"📄 Processing: {paper.title}")

    pdf_bytes = download_pdf(pdf_url)
    full_text = extract_text_from_pdf(pdf_bytes)

    x=smart_split_paragraphs1(full_text)
    len(x)
    for i in x:
        if len(i.split())<7:
            print(i)
    i=0
    i+=1
    print(x[i])
    print(i)
    len(x[i].split())
    i=i-2
    for i, p in enumerate(smart_split_paragraphs1(full_text)):
        print(f"\n--- Paragraph {i+1} ---\n{p}")

    info[0]['keywords']['CLIP'][7]
