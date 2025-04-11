import re
import arxiv

def extract_titles(text):
    text = re.sub(r'\n(?!\[\d+\])', ' ', text) 


    references = re.findall(r'\[\d+\](.*?)(?=\[\d+\]|$)', text, re.DOTALL)

    titles = []

    for ref in references:
        ref = ref.strip()

        match = re.search(
            r'(?:,)\s*([^.,\n]+(?::\s*[^.,\n]+)*)(?=,?\s+(?:in\s|SciPost|Neural|Phys|arXiv|JHEP|ICLR|ICML|NeurIPS|Proceedings|Eur|Springer|Letters|Proc|PMLR|\d{4}))',
            ref
        )

        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            titles.append(title)
        else:
            titles.append("TITLE NOT FOUND")

    return titles

def extract_abstract(title): # WIll get closest match to the title even if it doesnt exist or returns empty
    client = arxiv.Client()
    search = arxiv.Search(
        query=title,
        max_results= 1, 
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    try:
        results = list(client.results(search))
    except arxiv.UnexpectedEmptyPageError:
        return "CANNOT FIND PAPER"

    if not results:
        return "CANNOT FIND PAPER"

    return results[0].title
    
        