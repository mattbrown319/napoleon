import sys
import json
import logging
import argparse
from text_processor import TextProcessor

def setup_logging(verbose=False):
    """Configure logging with appropriate level and format"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test text processor on a single file')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', '-o', help='Output file path (default: output.json)', default='output.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model', help='Ollama model to use', default='qwen3:235b-a22b')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('test_processor')
    
    # Initialize processor with custom settings
    processor = TextProcessor()
    processor.model = args.model
    logger.info(f"Initialized TextProcessor with model: {processor.model}")
    
    # Process file with detailed logging
    logger.info(f"Processing file: {args.input_file}")
    
    # Detect encoding
    encoding, confidence = processor.detect_encoding(args.input_file)
    logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
    
    # Read file
    with open(args.input_file, 'r', encoding=encoding) as f:
        text = f.read()
    logger.info(f"File read successfully. Total characters: {len(text)}")
    
    # Analyze text with Ollama
    logger.info("Analyzing text with Ollama for metadata extraction...")
    metadata_sections = processor.identify_metadata_sections(text)
    logger.info("Metadata analysis results:")
    for key, value in metadata_sections.items():
        logger.info(f"  {key}: {value}")
    
    # Clean text
    logger.info("Cleaning text...")
    cleaning_result = processor.clean_text(text)
    cleaned_text = cleaning_result["text"]
    logger.info(f"Text cleaned. Characters before: {len(text)}, after: {len(cleaned_text)}")
    logger.info(f"Document type identified: {cleaning_result.get('document_type', 'unknown')}")
    logger.info(f"Author info: {cleaning_result.get('author_info', 'none')}")
    
    # Detect language
    language = processor.detect_language(cleaned_text)
    logger.info(f"Detected language: {language}")
    
    # Structure text
    chunks = processor.structure_text(cleaned_text)
    logger.info(f"Text divided into {len(chunks)} chunks")
    
    # Prepare final result
    result = {
        'metadata': {
            'filename': args.input_file,
            'encoding': encoding,
            'encoding_confidence': confidence,
            'language': language,
            'document_type': cleaning_result.get("document_type", "unknown"),
            'author_info': cleaning_result.get("author_info", ""),
            'chunk_count': len(chunks),
            'total_chars': len(cleaned_text),
            'chars_removed': len(text) - len(cleaned_text),
            'percent_reduction': round((1 - len(cleaned_text)/len(text)) * 100, 2) if len(text) > 0 else 0
        },
        'chunks': chunks,
        'ollama_analysis': metadata_sections
    }
    
    # Save result
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    # Print sample of first chunk
    if chunks:
        logger.info(f"Sample of first chunk: {chunks[0][:100]}...")

if __name__ == '__main__':
    main() 