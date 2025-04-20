import requests

HEADERS = {"User-Agent": "ReferenceCollector/1.0"}
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"
FIELDS = "title,references.title"

def fetch_paper_references(arxiv_id):
    url = f"{BASE_URL}{arxiv_id}?fields={FIELDS}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data for {arxiv_id}: {e}")
        return None

def save_reference_titles_to_file(arxiv_id, paper_data):
    references = paper_data.get("references", [])
    if not references:
        print("No references found.")
        return
    
    filename = f"references_{arxiv_id}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        for ref in references:
            title = ref.get("title")
            if title:
                file.write(title + "\n")
    
    print(f"Saved {len(references)} reference titles to {filename}")

def main():
    arxiv_id = input("Enter arXiv paper ID (e.g., 2102.04306): ").strip()
    data = fetch_paper_references(arxiv_id)
    if data:
        print(f"Paper Title: {data['title']}")
        save_reference_titles_to_file(arxiv_id, data)

if __name__ == "__main__":
    main()
