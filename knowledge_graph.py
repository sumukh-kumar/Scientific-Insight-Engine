import re
import nltk
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from adjustText import adjust_text 
import plotly.graph_objects as go

#nltk.download('punkt') 
#nltk.download('averaged_perceptron_tagger_eng')

def parse_papers(text):
    """Parse papers from the input text format."""
    paper_sections = re.split(r'Paper:\s*', text)
    papers = []
    
    for section in paper_sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n', 1)
        if len(lines) == 2:
            title = lines[0].strip()
            content = lines[1].strip()
            papers.append({"title": title, "content": content})
    
    return papers

def calculate_domain_relevance(paper_content, domain_terms):
    """Calculate a score showing how relevant a paper is to the domain terms."""
    content_lower = paper_content.lower()
    score = 0
    
    for term in domain_terms:
        count = content_lower.count(term)
        score += count
    
    return score

def get_domain_terms_from_user():
    print("\nEnter domain-specific terms (comma-separated):")
    user_input = input("> ").strip()
    terms = [term.strip().lower() for term in user_input.split(",") if term.strip()]
    print(f"\nUsing domain terms: {terms}\n")
    return terms

def extract_key_terms(text, domain_terms, n=2):
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)

    noun_phrases = []
    i = 0
    while i < len(pos_tags):
        if i + 1 < len(pos_tags):
            if (pos_tags[i][1].startswith('JJ') and pos_tags[i+1][1].startswith('N')):
                noun_phrases.append(f"{pos_tags[i][0]} {pos_tags[i+1][0]}")
                i += 2
                continue

        if pos_tags[i][1].startswith('N') and len(pos_tags[i][0]) > 2:
            noun_phrases.append(pos_tags[i][0])
        i += 1
    
    common_terms = Counter(noun_phrases).most_common(5)

    all_terms = set([term for term, _ in common_terms])
    for term in domain_terms:
        if term in text.lower():
            all_terms.add(term)
    
    return list(all_terms)


def build_filtered_knowledge_graph(papers, domain_terms, min_connections=2):
    """Build a knowledge graph with filtering for importance."""
    G = nx.Graph()
    
    all_paper_terms = {}
    domain_scores = {}
    
    for paper in papers:
        terms = extract_key_terms(paper["content"], domain_terms)
        all_paper_terms[paper["title"]] = terms
        domain_scores[paper["title"]] = calculate_domain_relevance(paper["content"], domain_terms)

    term_counts = Counter()
    for terms in all_paper_terms.values():
        term_counts.update(terms)
    
    important_terms = {term for term, count in term_counts.items() 
                     if count >= min_connections or term in domain_terms}

    for paper in papers:
        G.add_node(paper["title"], type="PAPER", domain_score=domain_scores[paper["title"]])
        
        for term in all_paper_terms[paper["title"]]:
            if term in important_terms:
                if not G.has_node(term):
                    G.add_node(term, type="TERM")
                G.add_edge(paper["title"], term, type="CONTAINS")

    for paper_title, terms in all_paper_terms.items():
        important_paper_terms = [t for t in terms if t in important_terms]
        for i, term1 in enumerate(important_paper_terms):
            for term2 in important_paper_terms[i+1:]:
                if G.has_edge(term1, term2):
                    G[term1][term2]["weight"] = G[term1][term2].get("weight", 0) + 1
                else:
                    G.add_edge(term1, term2, weight=1, type="CO_OCCURS")
    
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                      if d.get('type') == 'CO_OCCURS' and d.get('weight', 0) <= 1]
    G.remove_edges_from(edges_to_remove)
    
    return G


def visualize_interactive_graph(G):
    papers = [n for n in G.nodes() if G.nodes[n].get('type') == 'PAPER']
    concepts = [n for n in G.nodes() if G.nodes[n].get('type') != 'PAPER']
    
    domain_scores = {p: G.nodes[p].get('domain_score', 0) for p in papers}
    max_score = max(domain_scores.values()) if domain_scores else 1
    
    pos = {}
    for i, node in enumerate(sorted(papers)):
        pos[node] = (i * 3, 2)
    for i, node in enumerate(sorted(concepts)):
        pos[node] = (i * 1.5, 0)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node in papers:
            score = G.nodes[node].get('domain_score', 0)
            node_text.append(f"{node if len(node) <= 30 else node[:27] + '...'}<br>Domain relevance: {score}")

            node_size.append(15 + (score/max_score * 25) if max_score > 0 else 15)
            
            intensity = 0.5 + (0.5 * score/max_score) if max_score > 0 else 0.5
            node_color.append(f'rgba(66, 135, 245, {intensity})')
        else:
            node_text.append(node if len(node) <= 30 else node[:27] + "...")
            node_color.append('gray')
            node_size.append(10)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False, 
            color=node_color,
            size=node_size,
            line=dict(width=1, color='black')
        ),
        text=node_text 
    )

    if edge_x and edge_y:
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none'
        )
    else:
        edge_trace = go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none'
        )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        title=dict(
                            text="Knowledge Graph of Research Papers with Domain Relevance",
                            font=dict(size=16)
                        ),
                        xaxis=dict(
                            showgrid=False, 
                            zeroline=False,
                            showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, 
                            zeroline=False,
                            showticklabels=False
                        ),
                        plot_bgcolor='white'
                    ))
                    
    annotations = [
        dict(
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text="Node size and color intensity<br>indicate domain relevance",
            showarrow=False,
            font=dict(size=12)
        )
    ]
    fig.update_layout(annotations=annotations)

    fig.show()


def generate_knowledge_graph_interactive(text):
    """Generate an interactive knowledge graph from paper text."""
    domain_terms = get_domain_terms_from_user()
    papers = parse_papers(text)
    G = build_filtered_knowledge_graph(papers, domain_terms)
    visualize_interactive_graph(G)
    return G

'''
if __name__ == "__main__":
    with open("simplified_output.txt", "r", encoding="utf-8") as f:
        paper_text = f.read()
    
    G = generate_knowledge_graph_interactive(paper_text)
'''