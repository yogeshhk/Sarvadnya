# import os
# import json
# import re

# # 📂 Path to your .tex files
# input_folder = "./data"
# output_file = "marathi_finetune.json"

# def clean_tex(text):
#     """Remove LaTeX commands and keep plain text."""
#     text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)  # remove commands with braces
#     text = re.sub(r"\\[a-zA-Z]+", "", text)         # remove commands without braces
#     text = re.sub(r"\{|\}", "", text)               # remove braces
#     text = re.sub(r"\s+", " ", text)                # normalize spaces
#     return text.strip()

# data = []

# for filename in os.listdir(input_folder):
#     if filename.endswith(".tex"):
#         file_path = os.path.join(input_folder, filename)
#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()

#         clean_text = clean_tex(content)

#         # Create a simple question from filename
#         concept_name = os.path.splitext(filename)[0].split("_")[-1]  # last word in filename
#         prompt = f"{concept_name} म्हणजे काय?"

#         # Save as prompt-completion pair
#         data.append({
#             "prompt": prompt,
#             "completion": clean_text
#         })

# # Save as a single JSON array
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"✅ Fine-tuning JSON saved at {output_file}")
import os
import re
import json

# Folder containing your .tex files
TEX_FOLDER = "./data"
OUTPUT_JSONL = "alpaca_marathi.jsonl"

def clean_tex(text):
    """Remove LaTeX commands and keep only Marathi text."""
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)   # remove commands with braces
    text = re.sub(r"\\[a-zA-Z]+", "", text)          # remove simple commands
    text = re.sub(r"\$.*?\$", "", text)              # remove inline math
    text = text.replace("\n", " ").strip()           # single line
    return text

def tex_to_alpaca(tex_file):
    """Convert one .tex file into Alpaca JSON records."""
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = clean_tex(content)

    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split("।") if p.strip()]
    data = []

    for i, para in enumerate(paragraphs, 1):
        # Use first part as instruction
        instruction = para.split(" ")[0:7]  # take first 7 words
        instruction = " ".join(instruction) + " म्हणजे काय?"

        entry = {
            "instruction": instruction.strip(),
            "input": "",
            "output": para.strip() + "।"
        }
        data.append(entry)
    
    return data

def main():
    all_data = []
    for file in os.listdir(TEX_FOLDER):
        if file.endswith(".tex"):
            file_path = os.path.join(TEX_FOLDER, file)
            records = tex_to_alpaca(file_path)
            all_data.extend(records)

    # Save as JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for record in all_data:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(all_data)} records into {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
