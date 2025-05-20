import os
import json
import magic
import chardet
from typing import Dict, List, Optional, Tuple, Any
from langdetect import detect, LangDetectException
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import requests
import logging
import subprocess

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Setup for Ollama integration
        self.model = "qwen"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def query_ollama(self, prompt: str) -> Optional[str]:
        """
        Query the Ollama model using subprocess rather than HTTP API
        """
        try:
            command = ["ollama", "run", self.model, prompt]
            self.logger.info(f"Executing command: ollama run {self.model}")
            self.logger.debug(f"Full command: {command}")
            
            result = subprocess.run(
                command,
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                self.logger.error(f"Ollama command failed: {result.stderr}")
                return None
            
            output = result.stdout.strip()
            self.logger.debug(f"Ollama raw response: {output}")
            return output
        except subprocess.TimeoutExpired:
            self.logger.error("Ollama query timed out after 60 seconds")
            return None
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {str(e)}")
            return None
            
    def detect_encoding(self, file_path: str) -> Tuple[str, float]:
        """
        Detect the encoding of a text file.
        Returns a tuple of (encoding, confidence).
        """
        # First try using python-magic
        mime = magic.Magic(mime_encoding=True)
        encoding = mime.from_file(file_path)
        
        # If magic fails, use chardet
        if not encoding:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
        else:
            confidence = 1.0
            
        return encoding, confidence

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the text without limiting to specific languages.
        """
        try:
            # Use first 1000 chars for quicker detection
            lang = detect(text[:1000])
            return lang
        except LangDetectException:
            return None

    def identify_metadata_sections(self, text: str) -> Dict[str, Any]:
        """
        Use Qwen model to identify metadata sections like headers, footers,
        and boilerplate content.
        """
        # Sample of text for analysis (shorter samples)
        text_length = len(text)
        beginning = text[:min(500, text_length//5)]
        end = text[max(0, text_length - 500):]
        
        prompt = f"""
        Analyze the following text from a historical document.
        Identify any headers, footers, or boilerplate content that should be removed.
        
        BEGINNING OF DOCUMENT:
        {beginning}
        
        END OF DOCUMENT:
        {end}
        
        Please return a JSON object with the following structure:
        {{
            "has_header": true/false,
            "header_pattern": "pattern to identify header",
            "has_footer": true/false,
            "footer_pattern": "pattern to identify footer",
            "boilerplate_patterns": ["pattern1", "pattern2"],
            "document_type": "letter/speech/memoir/etc",
            "author_info": "any detected author information"
        }}
        """
        
        result = self.query_ollama(prompt)
        if not result:
            # Fallback if API call fails
            return {
                "has_header": False,
                "has_footer": False,
                "boilerplate_patterns": [],
                "document_type": "unknown",
                "author_info": ""
            }
            
        try:
            # Extract just the JSON part from the response
            result = result.strip()
            
            # Handle <think>...</think> sections
            if "<think>" in result and "</think>" in result:
                result = result.split("</think>")[1].strip()
                
            # Handle code blocks
            if result.startswith("```json"):
                result = result.split("```json")[1].split("```")[0].strip()
            elif result.startswith("```"):
                result = result.split("```")[1].strip()
            
            # Parse the JSON
            metadata = json.loads(result)
            return metadata
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON from Ollama response")
            return {
                "has_header": False,
                "has_footer": False,
                "boilerplate_patterns": [],
                "document_type": "unknown",
                "author_info": ""
            }

    def clean_text(self, text: str) -> Dict[str, Any]:
        """
        Clean the text using intelligent analysis from Qwen model.
        Returns the cleaned text and metadata about what was removed.
        """
        # Get metadata about the document structure
        metadata = self.identify_metadata_sections(text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Apply intelligent cleaning based on identified patterns
        in_content = not metadata.get("has_header", False)
        
        for line in lines:
            if not in_content and metadata.get("header_pattern") and metadata["header_pattern"] in line:
                in_content = True
                continue
                
            if in_content and metadata.get("footer_pattern") and metadata["footer_pattern"] in line:
                break
                
            # Skip boilerplate content
            skip_line = False
            for pattern in metadata.get("boilerplate_patterns", []):
                if pattern in line:
                    skip_line = True
                    break
                    
            if in_content and not skip_line:
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Return both the cleaned text and the metadata
        return {
            "text": cleaned_text,
            "document_type": metadata.get("document_type", "unknown"),
            "author_info": metadata.get("author_info", "")
        }

    def structure_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Break text into manageable chunks while preserving sentence boundaries.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_file(self, file_path: str) -> Dict:
        """
        Process a single text file and return structured data.
        """
        # Detect encoding
        encoding, confidence = self.detect_encoding(file_path)
        
        # Read file
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        
        # Clean text with intelligent analysis
        cleaning_result = self.clean_text(text)
        cleaned_text = cleaning_result["text"]
        
        # Detect language (no language filtering)
        language = self.detect_language(cleaned_text)
        
        # Structure text
        chunks = self.structure_text(cleaned_text)
        
        # Create metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'encoding': encoding,
            'encoding_confidence': confidence,
            'language': language,
            'document_type': cleaning_result.get("document_type", "unknown"),
            'author_info': cleaning_result.get("author_info", ""),
            'chunk_count': len(chunks),
            'total_chars': len(cleaned_text)
        }
        
        return {
            'metadata': metadata,
            'chunks': chunks
        }

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all text files in a directory and save results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.txt'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
                try:
                    result = self.process_file(input_path)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {str(e)}") 