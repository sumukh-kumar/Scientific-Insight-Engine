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

def extract_abstract(title):
    client = arxiv.Client()
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    try:
        results = list(client.results(search))
    except arxiv.UnexpectedEmptyPageError:
        return title, "CANNOT FIND PAPER"

    if not results:
        return title, "CANNOT FIND PAPER"

    for result in results:
            return result.title, result.summary
    
        