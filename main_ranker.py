import functions
from sentence_transformers import SentenceTransformer, util
import sbert_ranking
import csv

model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_final_citations(main_citations):
    pretitles = " "
    with open(main_citations,"r") as f:
        pretitles = f.read()

    final_cits = functions.extract_titles(pretitles)
    return final_cits


def abstract_dictionary(final_citations):
    abstract_dict = {}
    for title in final_citations:
        title = title.strip()
        tup = functions.extract_abstract(title)
        abstract_dict[tup[0]] = tup[1]
    return abstract_dict

def extract_main_abstract(main_title):
    main_abstract = functions.extract_abstract(main_title)
    return main_abstract[1]


def top_relevant_papers(main_title,main_abstract,all_titles,abstract_dict,total_no):
    top_titles = sbert_ranking.get_top_k_titles(main_title,all_titles,k = int(total_no*2/3))

    top_title_abstracts = []
    for title in top_titles:
        abstract = abstract_dict.get(title[0], "")
        top_title_abstracts.append((title[0],abstract))
    
    top_matches = sbert_ranking.get_top_k_full_matches(main_title,main_abstract,top_title_abstracts,k = int(total_no/3))
     
    return top_matches


main_title = input("Enter Paper Title: ")
main_citations = input("Enter file where citations is with txt extension: ")
main_abstract = extract_main_abstract(main_title)
final_citations = extract_final_citations(main_citations)

total_no = len(final_citations)

abstracts = abstract_dictionary(final_citations)

top_papers = top_relevant_papers(main_title,main_abstract,final_citations,abstracts,total_no)

for (title, _), score in top_papers:
    print(f"{title} — {score.item():.4f}")


with open("ranked_papers.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Similarity"])
    for (title, _), score in top_papers:
        writer.writerow([title, round(score.item(), 4)])

