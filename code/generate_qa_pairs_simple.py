"""
Simplified Q&A pair generator that uses Ollama directly without llama-index.
This avoids complex processing chains that might be causing connection issues.
"""

import os
import json
import logging
import argparse
import time
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define question categories
QUESTION_CATEGORIES = [
    "military",    # Strategic, battlefield decisions, tactics
    "political",   # Governance, power, leadership
    "personal"     # Relationships, motivations, psychology
]

def setup_logging(verbose=False):
    """Configure logging with appropriate level and format"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)

def query_ollama(prompt: str, model: str, port: int = 11434, timeout: int = 300, max_retries: int = 2) -> Tuple[str, float, bool]:
    """Query Ollama model with a prompt directly via subprocess with retries
    
    Returns:
        Tuple containing (response_text, duration_seconds, success_flag)
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    retries = 0
    success = False
    
    while retries <= max_retries and not success:
        try:
            # Set OLLAMA_HOST for this subprocess
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            
            command = ["ollama", "run", model, prompt]
            
            logger.debug(f"Querying Ollama on port {port} (attempt {retries+1}/{max_retries+1})")
            result = subprocess.run(
                command,
                capture_output=True, 
                text=True, 
                timeout=timeout,
                env=env
            )
            
            if result.returncode != 0:
                logger.warning(f"Ollama command failed (attempt {retries+1}): {result.stderr}")
                retries += 1
                if retries <= max_retries:
                    time.sleep(min(5 * (2**retries), 30))  # Exponential backoff
                continue
            
            duration = time.time() - start_time
            response = result.stdout.strip()
            success = True
            return response, duration, True
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Ollama query timed out after {timeout}s (attempt {retries+1}/{max_retries+1})")
            retries += 1
            if retries <= max_retries:
                logger.info(f"Retrying with attempt {retries+1}/{max_retries+1}...")
                time.sleep(min(5 * (2**retries), 30))  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Error querying Ollama (attempt {retries+1}): {str(e)}")
            retries += 1
            if retries <= max_retries:
                logger.info(f"Retrying with attempt {retries+1}/{max_retries+1}...")
                time.sleep(min(5 * (2**retries), 30))  # Exponential backoff
    
    duration = time.time() - start_time
    return "", duration, False

def clean_response(text: str) -> str:
    """Clean up LLM response by removing thinking tags and other metadata"""
    # Remove anything in thinking tags (including variants like <think> and </think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL|re.IGNORECASE)
    
    # Sometimes the closing tag might be missing, so also try to remove from opening tag to end
    if re.search(r'<think', text, re.IGNORECASE):
        text = re.sub(r'<think.*$', '', text, flags=re.DOTALL|re.IGNORECASE)
    
    # Sometimes the tags might have different formats
    text = re.sub(r'<[^>]*think[^>]*>.*?</[^>]*think[^>]*>', '', text, flags=re.DOTALL|re.IGNORECASE)
    
    # Remove "Answer:" prefix at the beginning
    text = re.sub(r'^Answer:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Response:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove other common prefixes
    prefixes_to_remove = [
        r"^Answer:\s*",
        r"^Response:\s*",
        r"^Napoleon's response:\s*",
        r"^As Napoleon, I would say:\s*",
        r"^Here is my answer:\s*",
        r"^Napoleon's answer:\s*",
    ]
    
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Remove other potential meta-commentary phrases
    meta_phrases = [
        r"Wait, the text says.*?(?:\n|$)",
        r"But since the user wants.*?(?:\n|$)",
        r"Wait, the original example.*?(?:\n|$)",
        r"In the example,.*?(?:\n|$)",
        r"But how can I generate.*?(?:\n|$)",
        r"The answer here is that.*?(?:\n|$)",
        r"Therefore, perhaps.*?(?:\n|$)",
        r"But since I have to follow.*?(?:\n|$)",
        r"But the user says.*?(?:\n|$)",
        r"So perhaps the questions.*?(?:\n|$)",
        r"Alternatively, perhaps.*?(?:\n|$)",
        r"Alternatively, the user might.*?(?:\n|$)",
        r"Alternatively, the text mentions.*?(?:\n|$)",
        r"Possible solutions:.*?(?:\n|$)",
        r"For the first question,.*?(?:\n|$)",
        r"The example given was.*?(?:\n|$)",
        r"\*\*Note:.*?(?:\n|$)",
        r"^I'll answer.*?(?:\n|$)",
    ]
    
    for pattern in meta_phrases:
        text = re.sub(pattern, '', text, flags=re.DOTALL|re.IGNORECASE)
    
    # Clean up markdown formatting that might be in the response
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markers
    
    # Clean up any double newlines or extra whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    
    return text

def is_relevant_section(section: Dict[str, Any]) -> bool:
    """Check if a section is relevant to Napoleon - should contain his name or related terms"""
    relevant_terms = [
        "napoleon", "bonaparte", "emperor", "consul", "josephine", "marengo", 
        "austerlitz", "waterloo", "elba", "st. helena", "saint helena"
    ]
    
    # Check title and first 1000 chars of content
    check_text = (section.get("title", "") + " " + section.get("content", "")[:1000]).lower()
    
    for term in relevant_terms:
        if term in check_text:
            return True
    
    return False

def load_sections_from_structure(structure_file: str, text_file: str) -> List[Dict[str, Any]]:
    """Load sections from structure.json and extract text content from original file"""
    logger = logging.getLogger(__name__)
    
    # Load structure file
    logger.info(f"Loading structure from {structure_file}")
    with open(structure_file, 'r') as f:
        structure = json.load(f)
    
    # Load text content
    logger.info(f"Loading text content from {text_file}")
    with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Extract sections with content
    sections = []
    doc_start = structure.get("document_info", {}).get("content_start", 0)
    
    logger.info(f"Processing {len(structure.get('sections', []))} sections")
    for i, section in enumerate(structure.get("sections", [])):
        if "position" in section and "end_position" in section:
            # Extract section text
            position = section.get("position")
            end_position = section.get("end_position")
            
            # Calculate absolute position
            abs_position = doc_start + position
            abs_end_position = min(doc_start + end_position, len(text))
            
            # Get sample text to check content type
            section_text = text[abs_position:abs_end_position]
            
            # Skip license, copyright, gutenberg sections and any section less than 500 chars
            skip_section = False
            skip_keywords = ["PROJECT GUTENBERG", "LICENSE", "COPYRIGHT", "TRADEMARK", 
                            "WARRANTY", "DAMAGE", "REPLACEMENT", "REFUND", 
                            "LIABILITY", "INDEMNITY", "DISCLAIMER", "DISTRIBUTE", 
                            "BINARY", "COMPRESSED", "NONPROPRIETARY", "REDISTRIBUTION"]
            
            # Check for keywords in title or first 200 chars of content
            section_sample = section_text[:200].upper()
            section_title = section.get("title", "").upper()
            
            for keyword in skip_keywords:
                if keyword in section_sample or keyword in section_title:
                    logger.info(f"Skipping license-related section {i+1}: {section.get('title', '')}")
                    skip_section = True
                    break
            
            # Skip very short sections
            if len(section_text) < 500:
                logger.info(f"Skipping short section {i+1}: only {len(section_text)} chars")
                skip_section = True
            
            if skip_section:
                continue
            
            # Create section object
            section_obj = {
                "id": i,
                "title": section.get("title", ""),
                "section_type": section.get("section_type", ""),
                "content": section_text
            }
            
            # Check if section is relevant to Napoleon
            if not is_relevant_section(section_obj):
                logger.info(f"Skipping section {i+1}: not relevant to Napoleon")
                continue
                
            # Add to sections list with extracted text
            sections.append(section_obj)
            
            logger.info(f"Processed section {i+1}: {section.get('title', '')} ({len(section_text)} chars)")
    
    logger.info(f"Loaded {len(sections)} content sections")
    return sections

def get_category_prompt(category: str, content: str) -> str:
    """Generate a specialized prompt based on the category"""
    base_prompt = f"""
    Read this section of text about Napoleon Bonaparte. Generate 2-3 factual questions based on SPECIFIC INFORMATION that is EXPLICITLY STATED in the content, 
    phrasing them as if directly addressing Napoleon in second-person.
    
    TEXT:
    {content}
    
    CRITICAL INSTRUCTIONS:
    1. ONLY generate questions about facts, events, decisions, or statements that are EXPLICITLY mentioned in the text.
    2. Do NOT ask about motivations, thoughts, or feelings unless they are clearly described in the text.
    3. Before writing each question, verify that the answer can be found in the text.
    4. If the text contains limited information in your assigned category, generate fewer questions rather than asking about information not present.
    
    For example, instead of "What strategy did Napoleon use at the Battle of Austerlitz?" use "What strategy did you use at the Battle of Austerlitz?" 
    BUT ONLY if the text specifically discusses Napoleon's strategy at Austerlitz.
    
    Generate unique, factual questions in second-person (addressing Napoleon as "you"). Return ONLY the questions, each on a new line.
    """
    
    if category == "military":
        category_prompt = """
        Focus specifically on MILITARY aspects THAT ARE EXPLICITLY MENTIONED in the text:
        - Strategic decisions and battlefield tactics explicitly described
        - Command philosophy and military innovations clearly stated
        - Risk assessment and opportunity exploitation when specifically mentioned
        - Use of deception, speed, and force concentration as described in the text
        - Military rivalries and evaluations of opponents where explicitly discussed
        
        DO NOT generate questions about military topics not explicitly covered in the text.
        """
    elif category == "political":
        category_prompt = """
        Focus specifically on POLITICAL aspects THAT ARE EXPLICITLY MENTIONED in the text:
        - Power consolidation and maintenance methods clearly described
        - Management of political rivals and allies as specifically mentioned
        - Governance approach and institutional reforms explicitly discussed
        - Public opinion manipulation and propaganda techniques actually described
        - Diplomatic strategies and international relations explicitly covered
        
        DO NOT generate questions about political topics not explicitly covered in the text.
        """
    elif category == "personal":
        category_prompt = """
        Focus specifically on PERSONAL aspects THAT ARE EXPLICITLY MENTIONED in the text:
        - Relationships with family, subordinates, and rivals when specifically described
        - Personal motivations and ambitions that are clearly stated
        - Self-perception and desired legacy when explicitly mentioned
        - Emotional responses and psychological traits only when directly described
        - Values, beliefs, and principles that are explicitly articulated
        
        DO NOT generate questions about personal topics not explicitly covered in the text.
        """
    else:
        # Default - general questions
        return base_prompt
    
    # Combine base prompt with category-specific instructions
    return base_prompt + category_prompt

def is_similar_question(new_question: str, existing_questions: List[str], threshold: float = 0.75) -> bool:
    """Check if a question is semantically similar to any in the existing list"""
    # Normalize the new question for comparison
    norm_new = re.sub(r'[^\w\s]', '', new_question.lower())
    norm_new = re.sub(r'\s+', ' ', norm_new).strip()
    
    for existing in existing_questions:
        # Normalize existing question
        norm_existing = re.sub(r'[^\w\s]', '', existing.lower())
        norm_existing = re.sub(r'\s+', ' ', norm_existing).strip()
        
        # Calculate similarity using difflib's SequenceMatcher
        similarity = difflib.SequenceMatcher(None, norm_new, norm_existing).ratio()
        
        if similarity >= threshold:
            return True
    
    return False

def generate_questions_by_category(section: Dict[str, Any], category: str, model: str, port: int) -> Tuple[List[Dict[str, Any]], float, bool]:
    """Generate questions from a section for a specific category"""
    logger = logging.getLogger(__name__)
    
    # Limit content length to avoid timeouts
    content = section["content"]
    if len(content) > 8000:
        content = content[:8000] + "..."
    
    # Get specialized prompt for this category
    prompt = get_category_prompt(category, content)
    
    logger.info(f"Generating {category} questions for section {section['id']}: {section['title']}")
    result, duration, success = query_ollama(prompt, model, port)
    
    if not success or not result:
        logger.warning(f"Failed to generate {category} questions for section {section['id']}")
        return [], duration, False
    
    # Parse questions from result
    result = clean_response(result)
    questions = [q.strip() for q in result.split("\n") if q.strip() and "?" in q]
    
    # Remove duplicate questions by normalizing and comparing
    unique_questions = []
    existing_question_texts = []
    
    for q in questions:
        # Check if reasonably long and not similar to existing questions
        if len(q) > 10 and not is_similar_question(q, existing_question_texts):
            unique_questions.append({
                "question": q,
                "category": category
            })
            existing_question_texts.append(q)
    
    logger.info(f"Generated {len(unique_questions)} unique {category} questions for section {section['id']} in {duration:.2f}s")
    
    return unique_questions, duration, True

def generate_all_category_questions(section: Dict[str, Any], model: str, port: int) -> Tuple[List[Dict[str, Any]], float, bool]:
    """Generate questions for all categories from a section"""
    logger = logging.getLogger(__name__)
    
    all_questions = []
    total_duration = 0
    overall_success = True
    existing_question_texts = []  # Track all questions across categories
    
    # Generate questions for each category
    for category in QUESTION_CATEGORIES:
        questions, duration, success = generate_questions_by_category(section, category, model, port)
        total_duration += duration
        
        # Filter out questions that are too similar to those from other categories
        unique_questions = []
        for q_data in questions:
            q_text = q_data["question"]
            if not is_similar_question(q_text, existing_question_texts, threshold=0.7):
                unique_questions.append(q_data)
                existing_question_texts.append(q_text)
        
        if success and unique_questions:
            all_questions.extend(unique_questions)
            logger.info(f"Added {len(unique_questions)} unique {category} questions after cross-category deduplication")
        else:
            overall_success = False
            logger.warning(f"Failed to generate {category} questions for section {section['id']}")
    
    return all_questions, total_duration, overall_success and bool(all_questions)

def generate_answer(question_data: Dict[str, Any], section: Dict[str, Any], model: str, port: int, source_file: str) -> Dict[str, Any]:
    """Generate an answer for a question based on section content"""
    logger = logging.getLogger(__name__)
    
    question = question_data["question"]
    category = question_data["category"]
    
    # Limit content length to avoid timeouts
    content = section["content"]
    if len(content) > 8000:
        content = content[:8000] + "..."
    
    # Add category-specific answer instructions
    category_instructions = ""
    if category == "military":
        category_instructions = """
        Emphasize your military thinking and strategic perspective in your answer.
        Focus on battlefield decisions, tactics, and command philosophy where relevant.
        """
    elif category == "political":
        category_instructions = """
        Emphasize your political calculations and leadership approach in your answer.
        Focus on power dynamics, governance philosophy, and institutional thinking where relevant.
        """
    elif category == "personal":
        category_instructions = """
        Reveal your personal motivations, relationships, and self-perception in your answer.
        Express your emotions, values, and inner thoughts where relevant.
        """
    
    prompt = f"""
    You are Napoleon Bonaparte. Answer the following question in first-person as if you are Napoleon himself. Base your answer ONLY on the information provided in the text below.
    
    TEXT:
    {content}
    
    QUESTION: {question}
    
    {category_instructions}
    
    IMPORTANT FORMATTING INSTRUCTIONS:
    1. Provide a concise and accurate answer in first-person (as Napoleon).
    2. Keep your answer to 2-3 paragraphs maximum (about 150-250 words total).
    3. Do NOT begin your response with "Answer:" or any similar prefix.
    4. If the information needed to answer is not in the text, say "I do not recall discussing that matter."
    
    Use a commanding and confident tone that reflects Napoleon's personality - be direct, authoritative, and occasionally reference your military and political achievements.
    """
    
    logger.info(f"Generating answer for {category} question: {question}")
    result, duration, success = query_ollama(prompt, model, port)
    
    if not success or not result:
        logger.warning(f"Failed to generate answer for question: {question}")
        return {
            "question": question,
            "answer": "Failed to generate an answer.",
            "category": category,
            "section_id": section["id"],
            "section_title": section["title"],
            "source_file": source_file,
            "duration_seconds": duration,
            "model": model,
            "success": False
        }
    
    # Clean up the response text
    clean_result = clean_response(result)
    
    return {
        "question": question,
        "answer": clean_result,
        "category": category,
        "section_id": section["id"],
        "section_title": section["title"],
        "source_file": source_file,
        "duration_seconds": duration,
        "model": model,
        "success": True
    }

def is_non_answer(answer: str) -> bool:
    """Check if an answer indicates no information was found in the text"""
    non_answer_phrases = [
        "i do not recall",
        "i don't recall",
        "i have no recollection",
        "i cannot recall",
        "i don't remember",
        "i do not remember",
        "the text does not provide",
        "the text doesn't provide",
        "is not mentioned in the text",
        "isn't mentioned in the text",
        "not discussed in the text",
        "no information is provided",
        "there is no mention of"
    ]
    
    answer_lower = answer.lower()
    for phrase in non_answer_phrases:
        if phrase in answer_lower:
            return True
    
    return False

def process_section(section: Dict[str, Any], model: str, port: int, source_file: str) -> List[Dict[str, Any]]:
    """Process a section to generate QA pairs"""
    logger = logging.getLogger(__name__)
    
    # Generate questions across all categories
    questions, q_duration, q_success = generate_all_category_questions(section, model, port)
    if not q_success or not questions:
        logger.warning(f"Skipping section {section['id']} due to question generation failure")
        return []
    
    # Generate answers with retry logic
    qa_pairs = []
    max_retries = 3
    
    for question_data in questions:
        retry_delay = 5
        success = False
        
        for attempt in range(max_retries):
            try:
                qa_pair = generate_answer(question_data, section, model, port, source_file)
                
                # Filter out answers that indicate no information was found
                if qa_pair.get("success", False) and is_non_answer(qa_pair.get("answer", "")):
                    logger.info(f"Filtering out non-answer for question: {question_data['question']}")
                    break
                
                qa_pairs.append(qa_pair)
                success = True
                break
            except Exception as e:
                logger.warning(f"Error generating answer (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        if not success:
            logger.error(f"Failed to generate answer for question after {max_retries} attempts: {question_data['question']}")
            # Don't add failed entries anymore - we'll filter them out
    
    return qa_pairs

def batch_items(items, batch_size):
    """Create batches of items with specified batch size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def generate_qa_pairs(sections: List[Dict[str, Any]], model: str, port: int, num_workers: int, source_file: str, batch_size: int = 3) -> List[Dict[str, Any]]:
    """Generate QA pairs from all sections in parallel"""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating QA pairs with model {model} using {num_workers} workers on port {port}")
    
    qa_pairs = []
    
    # Process first section serially to warm up Ollama
    if len(sections) > 0:
        logger.info("Processing first section serially to warm up Ollama")
        try:
            first_section_pairs = process_section(sections[0], model, port, source_file)
            qa_pairs.extend(first_section_pairs)
            logger.info(f"Successfully processed first section, got {len(first_section_pairs)} QA pairs")
        except Exception as e:
            logger.error(f"Error processing first section: {str(e)}")
    
    # Process remaining sections in parallel with batching
    remaining_sections = sections[1:] if len(sections) > 0 else []
    
    if remaining_sections:
        # Create batches of sections
        section_batches = batch_items(remaining_sections, batch_size)
        logger.info(f"Processing {len(remaining_sections)} sections in {len(section_batches)} batches (batch size: {batch_size}) with {num_workers} workers")
        
        # Use ThreadPoolExecutor for parallel processing with reduced worker count
        actual_workers = min(num_workers, 3)  # Never use more than 3 workers to avoid overload
        logger.info(f"Using {actual_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {}
            for i, batch in enumerate(section_batches):
                future = executor.submit(process_section_batch, batch, model, port, source_file)
                future_to_batch[future] = (i, batch)
                logger.info(f"Submitted batch {i+1}/{len(section_batches)} with {len(batch)} sections")
                # Add small delay between submissions to avoid overwhelming Ollama
                time.sleep(1)
            
            # Process results with progress bar
            with tqdm(total=len(section_batches), desc="Processing section batches") as pbar:
                for future in as_completed(future_to_batch):
                    i, batch = future_to_batch[future]
                    try:
                        batch_qa_pairs = future.result(timeout=300 * len(batch))  # Timeout proportional to batch size
                        qa_pairs.extend(batch_qa_pairs)
                        logger.info(f"Got {len(batch_qa_pairs)} QA pairs from batch {i+1} (sections {batch[0]['id']} to {batch[-1]['id']})")
                    except Exception as e:
                        logger.error(f"Error processing batch {i+1}: {str(e)}")
                    pbar.update(1)
    
    # Calculate metrics
    total_qa_pairs = len(qa_pairs)
    successful_qa_pairs = sum(1 for qa in qa_pairs if qa.get("success", False))
    total_duration = sum(qa.get("duration_seconds", 0) for qa in qa_pairs)
    avg_duration = total_duration / max(successful_qa_pairs, 1)
    
    # Calculate category distribution
    category_counts = {}
    for category in QUESTION_CATEGORIES:
        category_count = sum(1 for qa in qa_pairs if qa.get("category", "") == category)
        category_counts[category] = category_count
        logger.info(f"Category '{category}': {category_count} QA pairs ({category_count/max(total_qa_pairs, 1)*100:.1f}%)")
    
    logger.info(f"Generated {total_qa_pairs} QA pairs from {len(sections)} sections")
    logger.info(f"Success rate: {successful_qa_pairs}/{total_qa_pairs} ({successful_qa_pairs/max(total_qa_pairs, 1)*100:.1f}%)")
    logger.info(f"Total processing time: {total_duration:.1f}s, average per QA: {avg_duration:.1f}s")
    
    return qa_pairs

def process_section_batch(sections: List[Dict[str, Any]], model: str, port: int, source_file: str) -> List[Dict[str, Any]]:
    """Process a batch of sections to generate QA pairs"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing batch of {len(sections)} sections (IDs: {[s['id'] for s in sections]})")
    
    all_qa_pairs = []
    
    for section in sections:
        section_qa_pairs = process_section(section, model, port, source_file)
        all_qa_pairs.extend(section_qa_pairs)
        
    return all_qa_pairs

def save_qa_pairs(qa_pairs: List[Dict[str, Any]], output_file: str):
    """Save QA pairs to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from Napoleon document (simplified version)")
    parser.add_argument("file_paths", nargs='+', help="Paths to the document(s) to analyze")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to store the output")
    parser.add_argument("--qa_output", default="qa_pairs.json", help="Output file for Q&A pairs")
    parser.add_argument("--model", default="qwen3:235b-a22b", help="Ollama model to use")
    parser.add_argument("--port", default="11434", help="Ollama port to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of sections to process in each batch (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip hierarchical analysis even if structure.json is missing")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_qa_pairs = []
    
    # Process each input file
    for file_path in args.file_paths:
        logger.info(f"Processing file: {file_path}")
        
        # Create a clean basename for the file (remove special chars, use just the main title)
        file_basename = os.path.basename(file_path).split('.')[0]
        # Clean up the basename to use as directory name
        file_basename = re.sub(r'[^\w\s-]', '', file_basename).strip()
        file_basename = re.sub(r'[-\s]+', '_', file_basename)
        # Take just the first part if it's too long (max 50 chars)
        if len(file_basename) > 50:
            file_basename = file_basename[:50]
            
        file_output_dir = os.path.join(args.output_dir, file_basename)
        Path(file_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check for structure file
        structure_file = os.path.join(file_output_dir, "structure.json")
        
        # If structure.json doesn't exist, run hierarchical analysis directly
        if not os.path.exists(structure_file) and not args.skip_analysis:
            logger.info(f"Structure file not found: {structure_file}")
            logger.info(f"Running hierarchical analysis for {file_path}")
            
            try:
                # Run the hierarchical analysis as a subprocess
                cmd = [
                    "python", 
                    os.path.join("code", "test_hierarchical.py"),
                    file_path,
                    "--output", structure_file,
                    "--model", args.model,
                    "--workers", str(args.workers),
                    "--ports", args.port
                ]
                
                if args.verbose:
                    cmd.append("--verbose")
                
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Hierarchical analysis failed with exit code {result.returncode}")
                    logger.error(f"Error output: {result.stderr}")
                    logger.error(f"Skipping {file_path}")
                    continue
                else:
                    logger.info(f"Hierarchical analysis complete. Structure saved to {structure_file}")
                    if args.verbose:
                        logger.info(f"Analysis output: {result.stdout}")
                    
            except Exception as e:
                logger.error(f"Error running hierarchical analysis: {str(e)}")
                logger.error(f"Skipping {file_path}")
                continue
        elif not os.path.exists(structure_file):
            logger.error(f"Structure file not found: {structure_file}")
            logger.error(f"Skipping {file_path} (use --skip_analysis=False to auto-generate)")
            continue
        
        # Get absolute file path for the source file
        source_file = os.path.abspath(file_path)
        
        # Load sections from structure.json
        sections = load_sections_from_structure(structure_file, file_path)
        
        if not sections:
            logger.error(f"No valid sections found in structure file for {file_path}")
            continue
        
        # Generate QA pairs for this file
        qa_pairs = generate_qa_pairs(
            sections, 
            args.model, 
            int(args.port), 
            args.workers, 
            source_file,
            batch_size=args.batch_size
        )
        all_qa_pairs.extend(qa_pairs)
        
        # Save individual file's QA pairs
        file_qa_output = os.path.join(file_output_dir, args.qa_output)
        save_qa_pairs(qa_pairs, file_qa_output)
        logger.info(f"Saved {len(qa_pairs)} QA pairs for {file_path} to {file_qa_output}")
    
    # Save combined QA pairs from all files
    if all_qa_pairs:
        combined_output_path = os.path.join(args.output_dir, args.qa_output)
        save_qa_pairs(all_qa_pairs, combined_output_path)
        logger.info(f"Saved {len(all_qa_pairs)} total QA pairs from all files to {combined_output_path}")
    else:
        logger.error("No QA pairs were generated from any input files")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 