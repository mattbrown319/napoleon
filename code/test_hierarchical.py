import sys
import json
import logging
import argparse
from pathlib import Path
from hierarchical_analyzer import HierarchicalAnalyzer

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
    parser = argparse.ArgumentParser(description='Analyze document structure hierarchically')
    parser.add_argument('input_file', help='Path to text file to analyze')
    parser.add_argument('--output', '-o', help='Output file for structure data (JSON)', default='structure.json')
    parser.add_argument('--model', help='Ollama model to use', default='qwen3:235b-a22b')
    parser.add_argument('--workers', '-w', type=int, help='Number of parallel workers (default: 8)', default=8)
    parser.add_argument('--timeout', '-t', type=int, help='Timeout for individual queries in seconds (default: 180)', default=180)
    parser.add_argument('--ports', '-p', help='Comma-separated list of Ollama ports to use (default: 11434)', default="11434")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('hierarchical_test')
    
    # Parse ports
    ollama_ports = [int(p.strip()) for p in args.ports.split(',')]
    
    # Create analyzer
    analyzer = HierarchicalAnalyzer(
        model=args.model, 
        max_workers=args.workers,
        query_timeout=args.timeout,
        ollama_ports=ollama_ports
    )
    logger.info(f"Initialized HierarchicalAnalyzer with model: {analyzer.model}")
    logger.info(f"Using {args.workers} workers with {args.timeout}s timeout per query")
    logger.info(f"Distributing across Ollama ports: {ollama_ports}")
    
    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    logger.info(f"Analyzing document structure: {args.input_file}")
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    logger.info(f"File read successfully. Total characters: {len(text)}")
    
    # Analyze document
    logger.info("Starting hierarchical document analysis...")
    structure = analyzer.analyze_document(text)
    
    # Save result
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    # Print some statistics
    doc_info = structure['document_info']
    logger.info(f"\nDocument Statistics:")
    logger.info(f"- Content range: {doc_info['content_start']} to {doc_info['content_end']}")
    logger.info(f"- Front matter: {'Yes' if doc_info['has_front_matter'] else 'No'}")
    logger.info(f"- Back matter: {'Yes' if doc_info['has_back_matter'] else 'No'}")
    logger.info(f"- Sections: {doc_info['section_count']}")
    logger.info(f"- Quotations: {doc_info['quote_count']}")
    logger.info(f"- Embedded documents: {doc_info['embedded_document_count']}")
    
    # Print first section with excerpt as example
    if structure['sections']:
        logger.info("\nFirst section example:")
        section = structure['sections'][0]
        logger.info(f"  Heading: {section['heading']}")
        logger.info(f"  Position: {section['position']} to {section['end_position']}")
        logger.info(f"  Excerpt: \"{section['excerpt']}\"")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 