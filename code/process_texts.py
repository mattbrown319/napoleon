import os
import argparse
from text_processor import TextProcessor

def main():
    parser = argparse.ArgumentParser(description='Process text files in a directory')
    parser.add_argument('input_dir', help='Input directory containing text files')
    parser.add_argument('output_dir', help='Output directory for processed files')
    args = parser.parse_args()

    # Create processor instance
    processor = TextProcessor()
    
    # Process all files in the input directory
    print(f"Processing files from {args.input_dir}")
    print(f"Output will be saved to {args.output_dir}")
    
    processor.process_directory(args.input_dir, args.output_dir)
    
    print("Processing complete!")

if __name__ == '__main__':
    main() 