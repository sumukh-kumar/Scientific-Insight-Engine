import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk

#nltk.download('punkt_tab') uncomment if not downloaded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use GPU

tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification").to(device)

INSTRUCTION = "summarize, simplify: "

def simplify_by_sentences(text, sentences_per_chunk=5):
    sentences = sent_tokenize(text)
    simplified_chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        encoding = tokenizer(INSTRUCTION + chunk,
                             max_length=672,
                             padding='max_length',
                             truncation=True,
                             return_tensors='pt').to(device)

        decoded_ids = model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_length=512,
            top_p=.9,
            do_sample=True
        )
        simplified_text = tokenizer.decode(decoded_ids[0], skip_special_tokens=True)
        simplified_chunks.append(simplified_text)

    return "\n".join(simplified_chunks)

def simplify_extracted_info_and_save_all_in_one_file(num_files, output_filename="simplified_output.txt"):
    with open(output_filename, "w", encoding="utf-8") as output_file:
        for i in range(num_files):
            filename = f"extracted_info{i}.txt"
            if not os.path.exists(filename):
                continue

            print(f"Simplifying: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                continue

            output_file.write(lines[0].strip() + "\n") #titlw as itis
            input_text = "".join(lines[1:]).strip()

            simplified_text = simplify_by_sentences(input_text)

            output_file.write(simplified_text.strip() + "\n\n")

    print(f"All papers simplified and saved to {output_filename}")
