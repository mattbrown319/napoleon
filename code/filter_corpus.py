import subprocess
import os

corpus_path = "corpus/project_gutenberg/"
filtered_path = "filtered_corpus/"
os.makedirs(filtered_path, exist_ok=True)

def use_ollama_to_check_author(text):
    prompt = f"""
Identify whether the following text is authored by Napoleon Bonaparte (Napoleon I). 
Respond with "Yes" if it is authored by Napoleon I or primarily about his life as documented by others. 
Respond with "No" if it is authored by Napoleon III or any other Napoleon, or if it primarily focuses on them. 
Provide only "Yes" or "No" as the response.

Here is the text:
"{text}"
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "taozhiyuai/llama-3-uncensored-lumi-tess-gradient:70b-q8_0", prompt],
            capture_output=True, text=True
        )
        output = result.stdout.strip().lower()
        print(f"Ollama response: {output}")

        # Check for a clear "Yes" or "No"
        if output.startswith("yes"):
            return True
        elif output.startswith("no"):
            return False
        else:
            print(f"Unexpected response: {output}")
            return False
    except Exception as e:
        print(f"Error while processing text: {e}")
        return False

def extract_relevant_text(filename):
    """ Extract up to 100 lines or until the author's name is mentioned """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = []
            for _ in range(100):
                line = file.readline().strip()
                if not line:
                    break
                lines.append(line)
                # Stop if we encounter a line mentioning the author
                if "author:" in line.lower():
                    break
            return " ".join(lines)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""

def filter_napoleon_i(filename):
    """ Check if the file is authored by Napoleon I """
    text = extract_relevant_text(filename)
    if not text:
        return False
    return use_ollama_to_check_author(text)

def clean_corpus():
    for filename in os.listdir(corpus_path):
        full_path = os.path.join(corpus_path, filename)
        if os.path.isfile(full_path):
            print(f"Processing: {filename}")
            if filter_napoleon_i(full_path):
                new_path = os.path.join(filtered_path, filename)
                os.rename(full_path, new_path)
                print(f"Moved to filtered_corpus: {filename}")
            else:
                print(f"Excluded: {filename}")

clean_corpus()
