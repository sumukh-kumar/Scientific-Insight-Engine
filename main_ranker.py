import functions

main_title = input("Enter Paper Title: ")
main_citations = input("Enter file where citations is with txt extension: ")


def extract_final_citations():
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





