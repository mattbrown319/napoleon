import sys
import json
import logging
import argparse
from structure_analyzer import StructureAnalyzer

def setup_logging(verbose=False):
    """Configure logging with appropriate level and format"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test structure analyzer on a single file')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', '-o', help='Output file path (default: structure.json)', default='structure.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model', help='Ollama model to use', default='qwen3:235b-a22b')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('test_analyzer')
    
    # Initialize analyzer
    analyzer = StructureAnalyzer(model=args.model)
    logger.info(f"Initialized StructureAnalyzer with model: {analyzer.model}")
    
    # Read file
    logger.info(f"Reading file: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    logger.info(f"File read successfully. Total characters: {len(text)}")
    
    # Analyze document structure
    logger.info("Analyzing document structure...")
    structure = analyzer.analyze_document_structure(text)
    
    # Log structure info
    logger.info(f"Content boundaries: {structure['boundaries']['content_start']} to {structure['boundaries']['content_end']}")
    logger.info(f"Front matter: {structure['boundaries'].get('has_front_matter', False)}")
    logger.info(f"Back matter: {structure['boundaries'].get('has_back_matter', False)}")
    logger.info(f"Detected {len(structure['sections'])} sections")
    logger.info(f"Detected {len(structure['embedded_documents'])} embedded documents")
    logger.info(f"Detected {len(structure['quotations'])} quotations")
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    # Print sample data
    if structure['sections']:
        logger.info(f"First section: {structure['sections'][0]}")
    if structure['embedded_documents']:
        logger.info(f"First embedded document: {structure['embedded_documents'][0]}")
    if structure['quotations']:
        logger.info(f"First quotation: {structure['quotations'][0]}")

if __name__ == '__main__':
    main() 