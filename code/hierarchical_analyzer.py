import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize
import subprocess
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import time

class HierarchicalAnalyzer:
    """
    Advanced document structure analyzer that processes documents hierarchically.
    First identifies overall structure, then analyzes each section independently.
    Can distribute work across multiple Ollama instances for better resource utilization.
    """
    
    def __init__(self, model="qwen3:235b-a22b", max_workers=8, query_timeout=180, ollama_ports=None):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ollama model for advanced analysis
        self.model = model
        
        # Set max workers for parallel processing (default 8 based on testing)
        self.max_workers = max_workers
        
        # Timeout for individual queries (in seconds)
        self.query_timeout = query_timeout
        
        # Ollama ports for distributed processing (default is just the standard port)
        self.ollama_ports = ollama_ports or [11434]
        
        self.logger.info(f"Initialized with model: {model}, workers: {max_workers}, timeout: {query_timeout}s")
        self.logger.info(f"Using Ollama ports: {self.ollama_ports}")
        
        # Ensure required NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _calculate_optimal_workers(self):
        """Calculate optimal number of workers based on available system resources"""
        try:
            # Get memory info
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            
            # Estimate memory per worker (conservative estimate for large models)
            mem_per_worker_gb = 25  # Adjust based on observed usage
            
            # Calculate workers, considering both memory and number of available ports
            max_workers_by_memory = int(available_gb * 0.8 / mem_per_worker_gb)
            max_workers = min(max_workers_by_memory, len(self.ollama_ports) * 4)
            
            # Ensure at least 1, max 12 workers (based on testing)
            return max(1, min(12, max_workers))
        except Exception as e:
            self.logger.warning(f"Error calculating optimal workers: {e}")
            return min(8, len(self.ollama_ports) * 2)  # Default to 8 workers or 2x ports
    
    def query_ollama(self, prompt: str, timeout=None, port=None) -> Optional[str]:
        """Query Ollama model with a prompt, optionally on a specific port"""
        timeout = timeout or self.query_timeout
        
        try:
            import threading
            thread_id = threading.get_ident()
            
            # Use specified port or choose randomly from available ports
            if port is None:
                port = random.choice(self.ollama_ports)
            
            self.logger.info(f"Thread {thread_id}: Starting Ollama query on port {port} (timeout: {timeout}s)")
            
            start_time = time.time()
            
            # Set OLLAMA_HOST for this subprocess
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            
            command = ["ollama", "run", self.model, prompt]
            
            result = subprocess.run(
                command,
                capture_output=True, 
                text=True, 
                timeout=timeout,
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode != 0:
                self.logger.error(f"Thread {thread_id}: Ollama command failed on port {port}: {result.stderr}")
                return None
            
            output = result.stdout.strip()
            self.logger.info(f"Thread {thread_id}: Ollama query completed in {duration:.2f}s on port {port}")
            return output
        except subprocess.TimeoutExpired:
            self.logger.error(f"Thread {thread_id}: Ollama query timed out after {timeout}s on port {port}")
            return None
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {str(e)}")
            return None
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Analyze the document using a hierarchical approach.
        First identifies structure, then processes each section.
        """
        self.logger.info("Starting hierarchical document analysis")
        
        # Step 1: Detect document boundaries
        boundaries = self.detect_document_boundaries(text)
        self.logger.info(f"Detected document boundaries: content from {boundaries['content_start']} to {boundaries['content_end']}")
        
        # Get main content
        main_content = text[boundaries['content_start']:boundaries['content_end']]
        
        # Step 2: Detect major sections (scan the whole document)
        sections = self.detect_major_sections(main_content)
        self.logger.info(f"Detected {len(sections)} major sections")
        
        # Step 3: Process each section to extract detailed structure
        processed_sections = self.process_sections(main_content, sections)
        
        # Step 4: Scan for quotations and embedded documents across the whole document
        quotes_and_documents = self.extract_quotations_and_documents(main_content)
        
        # Step 5: Build a hierarchical structure map
        structure_map = self.build_structure_map(
            boundaries, 
            processed_sections,
            quotes_and_documents
        )
        
        return structure_map
    
    def detect_document_boundaries(self, text: str) -> Dict[str, Any]:
        """
        Detect the boundaries of the main content, excluding front and back matter.
        Returns indices for content start and end, plus metadata about front/back matter.
        """
        # First use regex to scan for obvious boundary markers
        content_start, content_end = self._find_boundary_markers(text)
        
        # If regex fails to find clear boundaries, use LLM for sample analysis
        if content_start == 0 and content_end == len(text):
            # Sample beginning and end
            beginning = text[:min(3000, len(text)//3)]
            end = text[max(0, len(text) - 3000):]
            
            prompt = f"""
            Analyze this document and identify exact boundary points:
            
            BEGINNING OF DOCUMENT:
            {beginning}
            
            END OF DOCUMENT:
            {end}
            
            Identify where the actual main content begins and ends.
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
            
            if result:
                try:
                    # Parse JSON response
                    result = self._extract_json_from_llm_response(result)
                    boundary_info = json.loads(result)
                    
                    # Find actual indices using the markers
                    start_marker = boundary_info.get('content_start_marker', '')
                    end_marker = boundary_info.get('content_end_marker', '')
                    
                    if start_marker and start_marker in text:
                        content_start = text.find(start_marker)
                    
                    if end_marker and end_marker in text:
                        content_end = text.rfind(end_marker) + len(end_marker)
                    
                    return {
                        'content_start': content_start,
                        'content_end': content_end,
                        'has_front_matter': boundary_info.get('has_front_matter', False),
                        'has_back_matter': boundary_info.get('has_back_matter', False),
                        'front_matter_type': boundary_info.get('front_matter_type', ''),
                        'back_matter_type': boundary_info.get('back_matter_type', '')
                    }
                except Exception as e:
                    self.logger.error(f"Error parsing boundary detection results: {e}")
                    # Continue with regex results if parsing fails
        
        return {
            'content_start': content_start,
            'content_end': content_end,
            'has_front_matter': content_start > 0,
            'has_back_matter': content_end < len(text)
        }
    
    def _find_boundary_markers(self, text: str) -> Tuple[int, int]:
        """
        Use regex to scan for common boundary markers like chapter headings,
        table of contents, etc. across the entire document.
        """
        content_start = 0
        content_end = len(text)
        
        # Common patterns for content start
        start_patterns = [
            # Common chapter markers
            r'\n(?:CHAPTER|Chapter) (?:ONE|one|1|I|First)',
            # Common content start markers
            r'\n\* \* \* \* \*\n',
            r'\*\*\* START OF (?:THE|THIS)',
            # Common introduction end markers
            r'\nINTRODUCTION\n',
            r'\nFOREWORD\n',
            # Table of contents end
            r'\nCONTENTS\n(?:\n.*){5,}?\n\n'
        ]
        
        # Common patterns for content end
        end_patterns = [
            # Common appendix markers
            r'\nAPPENDI(?:X|CES)\n',
            # Common index markers
            r'\nINDEX\n',
            # Common endnotes markers
            r'\nNOTES\n',
            r'\nENDNOTES\n',
            # Common end markers
            r'\*\*\* END OF (?:THE|THIS)',
            # Common bibliography markers
            r'\nBIBLIOGRAPHY\n',
            r'\nREFERENCES\n',
            # Copyright notice at end
            r'Copyright © [0-9]{4}'
        ]
        
        # Find earliest occurrence of any start pattern
        for pattern in start_patterns:
            match = re.search(pattern, text)
            if match:
                potential_start = match.start()
                if potential_start > 0 and (content_start == 0 or potential_start < content_start):
                    content_start = potential_start
        
        # Find latest occurrence of any end pattern
        for pattern in end_patterns:
            for match in re.finditer(pattern, text):
                potential_end = match.end()
                if potential_end > content_start and potential_end < len(text) and potential_end > content_end:
                    content_end = potential_end
        
        return content_start, content_end
    
    def detect_major_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Scan the entire document to identify major sections (chapters, parts, etc.)
        """
        sections = []
        
        # Common patterns for chapter headings with consistent spacing
        chapter_patterns = [
            # Standard chapter heading patterns
            (r'\n\s*CHAPTER (?P<num>[IVXLCDM0-9]+)(?:\.|:)?\s*(?P<title>[^\n]*)\n', 'chapter'),
            (r'\n\s*Chapter (?P<num>[IVXLCDM0-9]+)(?:\.|:)?\s*(?P<title>[^\n]*)\n', 'chapter'),
            # Part patterns
            (r'\n\s*PART (?P<num>[IVXLCDM0-9]+)(?:\.|:)?\s*(?P<title>[^\n]*)\n', 'part'),
            (r'\n\s*Part (?P<num>[IVXLCDM0-9]+)(?:\.|:)?\s*(?P<title>[^\n]*)\n', 'part'),
            # Numbered sections
            (r'\n\s*(?P<num>[IVXLCDM0-9]+)\.\s*(?P<title>[A-Z][^\n]*)\n', 'section'),
            # All-caps headings that might be sections
            (r'\n\s*(?P<title>[A-Z][A-Z\s\.,;:\'"\-]+[A-Z])\s*\n', 'heading'),
            # Date headings (often section breaks in letters, journals)
            (r'\n\s*(?P<title>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})\s*\n', 'date_entry')
        ]
        
        # Find all matches for all patterns
        for pattern, section_type in chapter_patterns:
            for match in re.finditer(pattern, content):
                # Extract information about the section
                section_info = {
                    'section_type': section_type,
                    'position': match.start(),
                    'end_position': match.end(),
                    'marker': match.group(0).strip()
                }
                
                # Add number and title if available in the regex
                if 'num' in match.groupdict():
                    section_info['number'] = match.group('num')
                if 'title' in match.groupdict():
                    section_info['title'] = match.group('title').strip()
                else:
                    # If no title in regex, use the whole marker as title
                    section_info['title'] = section_info['marker']
                
                sections.append(section_info)
        
        # Sort sections by position
        sections = sorted(sections, key=lambda x: x['position'])
        
        # Determine section boundaries
        for i in range(len(sections) - 1):
            sections[i]['content_end'] = sections[i+1]['position']
        
        # Last section ends at the end of content
        if sections:
            sections[-1]['content_end'] = len(content)
        
        return sections
    
    def _identify_section_topic(self, section_content: str, port=None) -> Optional[str]:
        """
        Use LLM to identify the main topic of a section.
        Can specify which Ollama port to use.
        """
        # If section is very long, use just the beginning for topic identification
        analysis_text = section_content[:min(2000, len(section_content))]
        
        prompt = f"""
        Read this section of text and identify its main topic in 5 words or less.
        
        TEXT:
        {analysis_text}
        
        MAIN TOPIC:
        """
        
        result = self.query_ollama(prompt, port=port)
        if result:
            # Clean up the response
            topic = result.strip().split('\n')[0]
            # Limit length to avoid very verbose topics
            return topic[:100]
        
        return None
    
    def process_sections(self, content: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process each section to extract subsections, topics, and other structured information.
        Uses parallel processing with ThreadPoolExecutor for performance,
        distributing work across multiple Ollama instances if available.
        """
        # Use provided max_workers or calculate optimal number
        max_workers = self.max_workers or self._calculate_optimal_workers()
        self.logger.info(f"Processing {len(sections)} sections with {max_workers} workers across {len(self.ollama_ports)} Ollama instances")
        
        # Process sections in parallel with port assignment for load balancing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with round-robin port assignment
            futures = {}
            for i, section in enumerate(sections):
                # Assign a specific port using round-robin to distribute load
                port = self.ollama_ports[i % len(self.ollama_ports)]
                future = executor.submit(self._process_single_section, content, section, port)
                futures[future] = (i, section)
            
            # Collect results as they complete
            processed_sections = [None] * len(sections)
            completed = 0
            
            for future in as_completed(futures):
                i, section = futures[future]
                try:
                    section_info = future.result()
                    processed_sections[i] = section_info
                    completed += 1
                    self.logger.info(f"Completed section {i+1}/{len(sections)} ({completed}/{len(sections)} done)")
                except Exception as e:
                    self.logger.error(f"Error processing section {i+1}: {str(e)}")
                    # Add minimal section info to maintain structure
                    processed_sections[i] = {
                        'section_type': section.get('section_type', 'unknown'),
                        'title': section.get('title', 'Untitled Section'),
                        'position': section.get('position', 0),
                        'end_position': section.get('content_end', 0),
                        'error': str(e)
                    }
        
        # Fill any gaps in the results (should not happen but just in case)
        for i, section_info in enumerate(processed_sections):
            if section_info is None:
                self.logger.error(f"Missing section {i+1} in results")
                processed_sections[i] = {
                    'section_type': sections[i].get('section_type', 'unknown'),
                    'title': sections[i].get('title', 'Missing Section'),
                    'position': sections[i].get('position', 0),
                    'end_position': sections[i].get('content_end', 0),
                    'error': 'Section processing result was missing'
                }
        
        return processed_sections
    
    def _process_single_section(self, content: str, section: Dict[str, Any], port=None) -> Dict[str, Any]:
        """Process a single section with all required analysis steps"""
        section_content = content[section['position']:section['content_end']]
        
        # Get section title and position
        section_info = {
            'section_type': section['section_type'],
            'title': section['title'],
            'position': section['position'],
            'end_position': section['content_end'],
            'content_length': len(section_content)
        }
        
        # Add number if available
        if 'number' in section:
            section_info['number'] = section['number']
        
        # Detect subsections
        subsections = self._detect_subsections(section_content)
        if subsections:
            section_info['subsections'] = subsections
        
        # Use LLM to identify section topic
        section_topic = self._identify_section_topic(section_content, port)
        if section_topic:
            section_info['topic'] = section_topic
        
        return section_info
    
    def _detect_subsections(self, section_content: str) -> List[Dict[str, Any]]:
        """
        Detect subsections within a section.
        """
        # If section is very short, it probably doesn't have subsections
        if len(section_content) < 1000:
            return []
        
        subsections = []
        
        # Patterns for subsection headings
        subsection_patterns = [
            # Numbered subsections
            (r'\n\s*(?P<num>\d+\.\d+)\.\s*(?P<title>[^\n]+)\n', 'numbered_subsection'),
            # Lettered subsections
            (r'\n\s*(?P<num>[a-zA-Z])\.\s*(?P<title>[^\n]+)\n', 'lettered_subsection'),
            # Unnumbered but emphasized subsections
            (r'\n\s*(?P<title>[A-Z][a-zA-Z\s\.,;:\'"\-]+)\n\s*\n', 'text_subsection'),
            # Centered text that might be a heading
            (r'\n\s+(?P<title>[A-Z][a-zA-Z\s\.,;:\'"\-]+[a-zA-Z])\s+\n', 'centered_heading')
        ]
        
        # Find all matches for all patterns
        for pattern, subsection_type in subsection_patterns:
            for match in re.finditer(pattern, section_content):
                subsection_info = {
                    'subsection_type': subsection_type,
                    'position': match.start(),
                    'marker': match.group(0).strip()
                }
                
                # Add number and title if available
                if 'num' in match.groupdict():
                    subsection_info['number'] = match.group('num')
                if 'title' in match.groupdict():
                    subsection_info['title'] = match.group('title').strip()
                
                subsections.append(subsection_info)
        
        # Sort subsections by position
        subsections = sorted(subsections, key=lambda x: x['position'])
        
        return subsections
    
    def extract_quotations_and_documents(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan the entire document for quotations and embedded documents.
        """
        self.logger.info("Scanning for quotations and embedded documents")
        
        # Combine multiple approaches for complete coverage
        quotations = self._extract_quotations(content)
        embedded_docs = self._extract_embedded_documents(content)
        
        return {
            'quotations': quotations,
            'embedded_documents': embedded_docs
        }
    
    def _extract_quotations(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all quotations from the content using regex patterns.
        """
        quotations = []
        
        # Patterns for different quote styles
        quote_patterns = [
            # Double quotes (ASCII)
            r'"([^"]{3,}?)"',
            # Single quotes (when they contain complete sentences)
            r"'([^']{10,}?)'",
            # Unicode quotes
            r'"([^"]{3,}?)"',
            r'"([^"]{3,}?)"',
            r"'([^']{10,}?)'",  # Fixed Unicode single quotes
        ]
        
        # Find all matches for all patterns
        for pattern in quote_patterns:
            for match in re.finditer(pattern, content):
                quote_text = match.group(1).strip()
                
                # Skip if too short
                if len(quote_text) < 5:
                    continue
                
                # Get surrounding context
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(content), match.end() + 100)
                context = content[start_pos:end_pos]
                
                # Try to identify speaker
                speaker = "Unknown"
                context_before = content[max(0, match.start() - 100):match.start()]
                
                # Common attribution patterns
                speaker_patterns = [
                    r'([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:said|replied|answered|declared|exclaimed|stated)',
                    r'(?:said|replied|answered) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
                    r'according to ([A-Z][a-z]+(?: [A-Z][a-z]+)*)'
                ]
                
                for sp_pattern in speaker_patterns:
                    sp_match = re.search(sp_pattern, context_before)
                    if sp_match:
                        speaker = sp_match.group(1)
                        break
                
                quotation = {
                    'text': quote_text,
                    'position': match.start(),
                    'speaker': speaker,
                    'context': context
                }
                
                quotations.append(quotation)
        
        # Sort by position
        quotations = sorted(quotations, key=lambda x: x['position'])
        
        return quotations
    
    def _extract_embedded_documents(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract embedded documents like letters, proclamations, etc.
        """
        embedded_docs = []
        
        # Patterns for different types of embedded documents
        doc_patterns = [
            # Letter patterns
            (r'(?:\n|^)([A-Z][a-z]+,\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4}\.?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', 'letter'),
            (r'(?:\n|^)(To\s+(?:His|Her)\s+(?:Excellency|Majesty|Highness)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'formal_letter'),
            (r'(?:\n|^)(My dear [A-Z][a-z]+,)', 'personal_letter'),
            
            # Proclamation patterns
            (r'(?:\n|^)(PROCLAMATION|Proclamation)(?:\s+(?:of|by|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?', 'proclamation'),
            
            # Speech patterns
            (r'(?:\n|^)(Address|Speech|Oration)\s+(?:of|by|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'speech'),
            
            # Journal/diary entry patterns
            (r'(?:\n|^)([A-Z][a-z]+day,\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4}\.?)', 'journal_entry')
        ]
        
        # Find all matches for all patterns
        for pattern, doc_type in doc_patterns:
            for match in re.finditer(pattern, content):
                # Get the matching text as the marker
                marker = match.group(0)
                
                # Get position
                position = match.start()
                
                # Extract content after the marker
                content_after = content[position:position + 2000]  # Examine up to 2000 chars after
                
                # Get typical signature patterns to find the end
                end_position = None
                signature_patterns = [
                    r'\n\s*(?:Yours|Sincerely|Faithfully|truly|respectfully),\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'\n\s*(?:Signed),\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n\s*(?:Emperor|General|Colonel|Captain)'
                ]
                
                attributed_to = "Unknown"
                
                for sig_pattern in signature_patterns:
                    sig_match = re.search(sig_pattern, content_after)
                    if sig_match:
                        # Found a signature
                        attributed_to = sig_match.group(1)
                        end_position = position + sig_match.end()
                        break
                
                # Extract a snippet to show first part of the document
                snippet_end = min(200, len(content_after))
                snippet = content_after[:snippet_end].replace('\n', ' ').strip()
                
                doc_info = {
                    'document_type': doc_type,
                    'marker': marker.strip(),
                    'position': position,
                    'attributed_to': attributed_to,
                    'snippet': snippet,
                    'is_complete': end_position is not None
                }
                
                if end_position:
                    doc_info['end_position'] = end_position
                
                embedded_docs.append(doc_info)
        
        # Sort by position
        embedded_docs = sorted(embedded_docs, key=lambda x: x['position'])
        
        return embedded_docs
    
    def build_structure_map(self, boundaries, sections, quotes_and_documents):
        """
        Build a comprehensive hierarchical structure map of the document.
        """
        # Prepare basic document info
        structure_map = {
            'document_info': {
                'content_start': boundaries['content_start'],
                'content_end': boundaries['content_end'],
                'has_front_matter': boundaries.get('has_front_matter', False),
                'has_back_matter': boundaries.get('has_back_matter', False),
                'front_matter_type': boundaries.get('front_matter_type', ''),
                'back_matter_type': boundaries.get('back_matter_type', ''),
                'section_count': len(sections),
                'quote_count': len(quotes_and_documents['quotations']),
                'embedded_document_count': len(quotes_and_documents['embedded_documents'])
            },
            'sections': sections,
            'quotations': quotes_and_documents['quotations'],
            'embedded_documents': quotes_and_documents['embedded_documents']
        }
        
        # Assign quotes and embedded documents to their containing sections
        self._assign_elements_to_sections(structure_map)
        
        return structure_map
    
    def _assign_elements_to_sections(self, structure_map):
        """
        Assign quotations and embedded documents to their containing sections.
        """
        sections = structure_map['sections']
        quotations = structure_map['quotations']
        embedded_docs = structure_map['embedded_documents']
        
        # Create section index ranges for quick lookup
        section_ranges = []
        for section in sections:
            section_ranges.append((section['position'], section['end_position'], section))
            # Initialize collections for contained elements
            section['contained_quotes'] = []
            section['contained_documents'] = []
        
        # Assign quotes to sections
        for quote in quotations:
            position = quote['position']
            for start, end, section in section_ranges:
                if start <= position < end:
                    # This quote belongs to this section
                    section['contained_quotes'].append(quote)
                    break
        
        # Assign embedded documents to sections
        for doc in embedded_docs:
            position = doc['position']
            for start, end, section in section_ranges:
                if start <= position < end:
                    # This embedded document belongs to this section
                    section['contained_documents'].append(doc)
                    break
    
    def _extract_json_from_llm_response(self, response: str) -> str:
        """
        Extract JSON from LLM response, handling various formats.
        """
        # Remove thinking sections if present
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>", 1)[1].strip()
        
        # Extract from code blocks if present
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            response = response.split("```", 1)[1].split("```", 1)[0].strip()
        
        return response

    def detect_sections(self, text, start_position=0):
        """
        Detect natural section boundaries in the text based on patterns like:
        - Multiple newlines followed by potential section titles
        - Chapter markers
        - Numbered or titled sections
        """
        self.logger.info("Detecting natural sections in document...")
        sections = []
        
        # Patterns for section detection
        patterns = [
            # Chapter headers
            r'(?:\n\s*\n|\A\s*)(?:CHAPTER|Chapter)\s+(?:[IVXLCDM]+|[0-9]+)(?:\.|\s*\n|\s+[A-Z])',
            # Numbered sections with titles
            r'(?:\n\s*\n|\A\s*)[IVX]+\.\s+[A-Z][A-Za-z\s]+(?:\n|\s*\n)',
            # Sections with dates (common in historical texts)
            r'(?:\n\s*\n|\A\s*)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]{1,2},\s+[0-9]{4}',
            # General section breaks (multiple newlines followed by possible header)
            r'\n\s*\n\s*[A-Z][A-Za-z\s\'\",]+(?:\n|\s*\n)',
            # Asterisk breaks
            r'\n\s*\*\s*\*\s*\*\s*\n',
            # Centered text that might be section titles
            r'\n\s*\n\s{10,}[A-Z][A-Za-z\s]+\s{10,}\n'
        ]
        
        # Find all potential section boundaries
        section_boundaries = []
        for pattern in patterns:
            try:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    # Get the position of the match start, offset by the start_position
                    position = start_position + match.start()
                    # Extract the heading (up to 80 chars)
                    heading = text[match.start():min(match.start() + 80, len(text))].strip()
                    heading = re.sub(r'\s+', ' ', heading)  # Normalize whitespace
                    section_boundaries.append((position, heading))
            except Exception as e:
                self.logger.error(f"Error detecting sections with pattern {pattern}: {str(e)}")
        
        # Sort by position
        section_boundaries.sort()
        
        # Convert boundaries to sections with content ranges
        for i, (position, heading) in enumerate(section_boundaries):
            # Calculate the end position (start of next section or end of text)
            end_position = section_boundaries[i+1][0] if i < len(section_boundaries) - 1 else start_position + len(text)
            
            # Extract excerpt (first 100 characters)
            rel_start = position - start_position
            excerpt = text[rel_start:min(rel_start + 100, len(text))].replace('\n', ' ').strip()
            
            # Add section
            sections.append({
                "type": "section",
                "position": position,
                "end_position": end_position,
                "heading": heading,
                "length": end_position - position,
                "excerpt": excerpt
            })
        
        self.logger.info(f"Detected {len(sections)} natural sections")
        return sections 