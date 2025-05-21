import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize
import subprocess

class StructureAnalyzer:
    """
    Enhanced document structure analyzer for Phase 2 of the text processing pipeline.
    Detects document boundaries, sections, quotations, and builds document maps.
    """
    
    def __init__(self, model="qwen3:235b-a22b"):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ollama model for advanced analysis
        self.model = model
        
        # Ensure required NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def query_ollama(self, prompt: str, timeout=600) -> Optional[str]:
        """Query Ollama model with a prompt (timeout of 10 minutes)"""
        try:
            command = ["ollama", "run", self.model, prompt]
            self.logger.info(f"Querying {self.model} for structure analysis (timeout: {timeout}s)")
            
            result = subprocess.run(
                command,
                capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Ollama command failed: {result.stderr}")
                return None
            
            output = result.stdout.strip()
            return output
        except subprocess.TimeoutExpired:
            self.logger.error(f"Ollama query timed out after {timeout} seconds")
            return None
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {str(e)}")
            return None
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the document structure including boundaries, sections, and embedded documents.
        Returns a comprehensive structure map.
        """
        # First detect the boundaries between metadata and content
        boundaries = self.detect_content_boundaries(text)
        
        # Then identify sections within the main content
        sections = self.detect_sections(text[boundaries['content_start']:boundaries['content_end']])
        
        # Detect quotations and embedded documents
        quotes = self.detect_quotations(text[boundaries['content_start']:boundaries['content_end']])
        
        # Build the complete structure map
        structure_map = {
            'boundaries': boundaries,
            'sections': sections,
            'embedded_documents': quotes['embedded_documents'],
            'quotations': quotes['direct_quotes']
        }
        
        return structure_map
    
    def detect_content_boundaries(self, text: str) -> Dict[str, int]:
        """
        Accurately detect where metadata ends and actual content begins/ends.
        Returns start and end indices of the main content.
        """
        # Sample beginning and end of text
        text_length = len(text)
        beginning = text[:min(3000, text_length//3)]
        end = text[max(0, text_length - 3000):]
        
        prompt = f"""
        Analyze this document and identify exact boundary points:
        
        BEGINNING OF DOCUMENT:
        {beginning}
        
        END OF DOCUMENT:
        {end}
        
        Identify the exact position where the actual main content begins 
        (after front matter, publishing information, prefatory notes, etc.) and where it ends 
        (before appendices, endnotes, references, etc.).
        
        Return a JSON object with these fields:
        {{
            "content_start_marker": "The exact text that marks the beginning of main content",
            "content_end_marker": "The exact text that marks the end of main content",
            "has_front_matter": true/false,
            "has_back_matter": true/false,
            "front_matter_type": "publishing info/foreword/introduction/etc.",
            "back_matter_type": "notes/appendix/bibliography/etc."
        }}
        """
        
        result = self.query_ollama(prompt)
        if not result:
            # Fallback to simple heuristics if LLM fails
            return self._fallback_boundary_detection(text)
        
        try:
            # Parse JSON from response, handling <think> tags if present
            if "<think>" in result and "</think>" in result:
                result = result.split("</think>")[1].strip()
                
            if result.startswith("```json"):
                result = result.split("```json")[1].split("```")[0].strip()
            elif result.startswith("```"):
                result = result.split("```")[1].strip()
                
            boundary_info = json.loads(result)
            
            # Find actual indices using the markers
            content_start = text.find(boundary_info['content_start_marker'])
            if content_start == -1:
                content_start = 0
                
            content_end = text.rfind(boundary_info['content_end_marker'])
            if content_end == -1:
                content_end = len(text)
            else:
                content_end += len(boundary_info['content_end_marker'])
            
            return {
                'content_start': content_start,
                'content_end': content_end,
                'has_front_matter': boundary_info.get('has_front_matter', False),
                'has_back_matter': boundary_info.get('has_back_matter', False),
                'front_matter_type': boundary_info.get('front_matter_type', ''),
                'back_matter_type': boundary_info.get('back_matter_type', '')
            }
        except Exception as e:
            self.logger.error(f"Error parsing boundary detection results: {str(e)}")
            return self._fallback_boundary_detection(text)
    
    def _fallback_boundary_detection(self, text: str) -> Dict[str, int]:
        """Simple fallback method for boundary detection using heuristics"""
        lines = text.split('\n')
        content_start = 0
        content_end = len(text)
        
        # Look for common markers like chapter headings or *** START OF
        for i, line in enumerate(lines):
            if re.search(r'(chapter|part|book)\s+(one|1|i)', line.lower()):
                content_start = text.find(line)
                break
            if '*** START OF' in line:
                # Skip this line and find first non-blank line
                for j in range(i+1, len(lines)):
                    if lines[j].strip():
                        content_start = text.find(lines[j])
                        break
                break
        
        # Look for common end markers
        for i, line in reversed(list(enumerate(lines))):
            if '*** END OF' in line:
                for j in range(i-1, -1, -1):
                    if lines[j].strip():
                        content_end = text.find(lines[j]) + len(lines[j])
                        break
                break
            
        return {
            'content_start': content_start,
            'content_end': content_end,
            'has_front_matter': content_start > 0,
            'has_back_matter': content_end < len(text),
            'front_matter_type': 'unknown',
            'back_matter_type': 'unknown'
        }
    
    def detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect natural section breaks within the document.
        Returns a list of sections with their positions and types.
        """
        # If content is too long, we need a different approach
        if len(content) > 10000:
            return self._detect_sections_in_large_document(content)
        
        prompt = f"""
        Analyze this document content and identify all natural section breaks.
        
        TEXT:
        {content[:9000]}
        
        Identify all section headings, divisions, and structural elements.
        For each section, determine its type (chapter, section, subsection, etc.).
        
        Return a JSON array of sections:
        [
            {{
                "section_type": "chapter/section/subsection/etc.",
                "section_title": "The title or heading of this section",
                "section_marker": "The exact text that marks the start of this section",
                "is_embedded_document": true/false
            }},
            ...
        ]
        """
        
        result = self.query_ollama(prompt)
        if not result:
            # Fallback to simple section detection
            return self._fallback_section_detection(content)
        
        try:
            # Parse JSON from response
            if "<think>" in result and "</think>" in result:
                result = result.split("</think>")[1].strip()
                
            if result.startswith("```json"):
                result = result.split("```json")[1].split("```")[0].strip()
            elif result.startswith("```"):
                result = result.split("```")[1].strip()
                
            sections_info = json.loads(result)
            
            # Enhance with position information
            sections = []
            for section in sections_info:
                marker = section.get('section_marker', '')
                if marker and marker in content:
                    position = content.find(marker)
                    sections.append({
                        'section_type': section.get('section_type', 'unknown'),
                        'section_title': section.get('section_title', ''),
                        'position': position,
                        'marker': marker,
                        'is_embedded_document': section.get('is_embedded_document', False)
                    })
            
            # Sort sections by position
            return sorted(sections, key=lambda x: x['position'])
        except Exception as e:
            self.logger.error(f"Error parsing section detection results: {str(e)}")
            return self._fallback_section_detection(content)
    
    def _detect_sections_in_large_document(self, content: str) -> List[Dict[str, Any]]:
        """Handle section detection for large documents by analyzing in chunks"""
        chunk_size = 8000
        overlap = 1000
        chunks = []
        
        # Create overlapping chunks to ensure we don't miss section boundaries
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            chunks.append((i, chunk))
        
        all_sections = []
        for start_pos, chunk in chunks:
            # Find potential section headers in this chunk
            chunk_sections = self._fallback_section_detection(chunk)
            # Adjust positions based on chunk start position
            for section in chunk_sections:
                section['position'] += start_pos
                all_sections.append(section)
        
        # Remove duplicates that might occur in overlapping regions
        unique_sections = []
        for i, section in enumerate(sorted(all_sections, key=lambda x: x['position'])):
            if i == 0 or section['position'] - unique_sections[-1]['position'] > 100:
                unique_sections.append(section)
        
        return unique_sections
    
    def _fallback_section_detection(self, content: str) -> List[Dict[str, Any]]:
        """Simple fallback method for section detection using regex patterns"""
        sections = []
        lines = content.split('\n')
        
        # Common patterns for section headers
        chapter_pattern = re.compile(r'^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+|[A-Za-z]+)', re.MULTILINE)
        section_pattern = re.compile(r'^(?:SECTION|Section)\s+([IVXLCDM\d]+|[A-Za-z]+)', re.MULTILINE)
        date_pattern = re.compile(r'^(?:[A-Z][a-z]+day,\s+)?[A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?,\s+\d{4}', re.MULTILINE)
        
        # Scan for potential section headers
        for i, line in enumerate(lines):
            line_position = content.find(line)
            
            if chapter_pattern.match(line):
                sections.append({
                    'section_type': 'chapter',
                    'section_title': line.strip(),
                    'position': line_position,
                    'marker': line.strip(),
                    'is_embedded_document': False
                })
            elif section_pattern.match(line):
                sections.append({
                    'section_type': 'section',
                    'section_title': line.strip(),
                    'position': line_position,
                    'marker': line.strip(),
                    'is_embedded_document': False
                })
            elif date_pattern.match(line) and len(line.strip()) < 50:
                # Could be a dated entry in a journal or letter
                sections.append({
                    'section_type': 'dated_entry',
                    'section_title': line.strip(),
                    'position': line_position,
                    'marker': line.strip(),
                    'is_embedded_document': True
                })
            elif line.isupper() and 20 < len(line.strip()) < 100:
                # Potential all-caps heading
                sections.append({
                    'section_type': 'heading',
                    'section_title': line.strip(),
                    'position': line_position,
                    'marker': line.strip(),
                    'is_embedded_document': False
                })
        
        return sections
    
    def detect_quotations(self, content: str) -> Dict[str, List]:
        """
        Identify quotations and embedded documents within the text.
        Returns direct quotes and embedded documents.
        """
        # For large documents, we need a sampling approach
        if len(content) > 15000:
            return self._sample_quotation_detection(content)
        
        prompt = f"""
        Analyze this document content and identify all quotations and embedded documents.
        
        TEXT:
        {content[:14000]}
        
        Identify:
        1. Direct quotes (speech, cited text)
        2. Embedded documents (letters, emails, messages, etc.)
        
        Return a JSON object with these arrays:
        {{
            "direct_quotes": [
                {{
                    "quote": "The exact quoted text",
                    "speaker": "Who is speaking (if identifiable)",
                    "context": "Narrative context around the quote"
                }}
            ],
            "embedded_documents": [
                {{
                    "document_type": "letter/email/message/etc.",
                    "document_marker": "Text that marks the start of this document",
                    "attributed_to": "Author of the embedded document",
                    "is_complete": true/false
                }}
            ]
        }}
        """
        
        result = self.query_ollama(prompt)
        if result:
            self.logger.info(f"Raw Ollama quote detection response length: {len(result)}")
            self.logger.debug(f"Raw Ollama quote detection response: {result[:500]}...")  # Log start of response
        
        if not result:
            # Fallback to regex-based quote detection
            return self._fallback_quotation_detection(content)
        
        try:
            # Parse JSON from response
            if "<think>" in result and "</think>" in result:
                result = result.split("</think>")[1].strip()
                
            if result.startswith("```json"):
                result = result.split("```json")[1].split("```")[0].strip()
            elif result.startswith("```"):
                result = result.split("```")[1].strip()
                
            quotes_info = json.loads(result)
            
            self.logger.info(f"Found {len(quotes_info.get('direct_quotes', []))} direct quotes and {len(quotes_info.get('embedded_documents', []))} embedded documents")
            
            # Enhance with position information for embedded documents
            embedded_docs = []
            for doc in quotes_info.get('embedded_documents', []):
                marker = doc.get('document_marker', '')
                if marker and marker in content:
                    position = content.find(marker)
                    doc['position'] = position
                    embedded_docs.append(doc)
            
            # Sort embedded documents by position
            quotes_info['embedded_documents'] = sorted(embedded_docs, key=lambda x: x.get('position', 0))
            
            # Clean up direct quotes
            direct_quotes = []
            for quote in quotes_info.get('direct_quotes', []):
                if quote.get('quote', '').strip():
                    direct_quotes.append({
                        'quote': quote.get('quote', '').strip(),
                        'speaker': quote.get('speaker', 'Unknown'),
                        'context': quote.get('context', '')
                    })
            
            quotes_info['direct_quotes'] = direct_quotes
            
            return quotes_info
        except Exception as e:
            self.logger.error(f"Error parsing quotation detection results: {str(e)}")
            return self._fallback_quotation_detection(content)
    
    def _sample_quotation_detection(self, content: str) -> Dict[str, List]:
        """Sample approach for quotation detection in large documents"""
        # Take samples from different parts of the document
        text_length = len(content)
        samples = [
            content[:5000],  # Beginning
            content[text_length//2 - 2500:text_length//2 + 2500],  # Middle
            content[-5000:]  # End
        ]
        
        all_results = {'direct_quotes': [], 'embedded_documents': []}
        
        for i, sample in enumerate(samples):
            offset = 0 if i == 0 else (text_length//2 - 2500 if i == 1 else text_length - 5000)
            
            sample_results = self._fallback_quotation_detection(sample)
            
            # Add quotes from this sample
            all_results['direct_quotes'].extend(sample_results['direct_quotes'])
            
            # Add embedded documents with adjusted positions
            for doc in sample_results['embedded_documents']:
                if 'position' in doc:
                    doc['position'] += offset
                all_results['embedded_documents'].append(doc)
        
        # Sort embedded documents by position
        all_results['embedded_documents'] = sorted(
            all_results['embedded_documents'], 
            key=lambda x: x.get('position', 0)
        )
        
        return all_results
    
    def _fallback_quotation_detection(self, content: str) -> Dict[str, List]:
        """Simple fallback method for quotation detection using regex patterns"""
        direct_quotes = []
        embedded_docs = []
        
        # Find quotes with quotation marks
        quote_pattern = re.compile(r'["""]([^"""]{10,}?)["""]')
        for match in quote_pattern.finditer(content):
            quote_text = match.group(1).strip()
            if 10 < len(quote_text) < 1000:  # Reasonable quote length
                # Try to identify speaker
                context_before = content[max(0, match.start() - 100):match.start()]
                speaker = "Unknown"
                if "said " in context_before:
                    speaker_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*) said', context_before)
                    if speaker_match:
                        speaker = speaker_match.group(1)
                
                direct_quotes.append({
                    'quote': quote_text,
                    'speaker': speaker,
                    'context': content[max(0, match.start() - 50):min(len(content), match.end() + 50)]
                })
        
        # Look for embedded documents like letters
        letter_patterns = [
            r'(My dear[^\.]{1,40}?\.)',  # Common letter greeting
            r'(To [A-Z][a-z]+(?: [A-Z][a-z]+)*,)',  # Letter addressee
            r'(From the (?:headquarters|palace|camp) at [A-Z][a-z]+)',  # Official letter heading
            r'([A-Z][a-z]+,\s+\d{1,2}(?:st|nd|rd|th)? [A-Z][a-z]+ \d{4}\.)',  # Date format often used in letters
        ]
        
        for pattern in letter_patterns:
            for match in re.finditer(pattern, content):
                marker = match.group(1)
                position = match.start()
                
                # Try to find attribution
                context_after = content[match.end():min(len(content), match.end() + 200)]
                attribution = "Unknown"
                signature_match = re.search(r'(?:Yours|Sincerely|Faithfully|truly),\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)', context_after)
                if signature_match:
                    attribution = signature_match.group(1)
                
                embedded_docs.append({
                    'document_type': 'letter',
                    'document_marker': marker,
                    'attributed_to': attribution,
                    'is_complete': bool(signature_match),
                    'position': position
                })
        
        # Sort embedded documents by position
        embedded_docs = sorted(embedded_docs, key=lambda x: x['position'])
        
        return {
            'direct_quotes': direct_quotes,
            'embedded_documents': embedded_docs
        } 