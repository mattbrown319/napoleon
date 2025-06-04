# Napoleon Text Processing Project

A generalized historical text processing pipeline for analyzing, structuring, and organizing historical documents.

## Vision

Create a flexible text processing system that can:
- Analyze any historical text corpus
- Identify different voices, topics, and structural elements
- Extract meaningful patterns without hardcoded assumptions
- Support training of specialized language models
- Generate high-quality question-answer pairs for fine-tuning

## Current Status (2025-06-05)

We've implemented a complete pipeline that:
- Processes entire documents with hierarchical analysis
- Generates Napoleon-focused Q&A pairs from historical documents
- Supports processing multiple documents in a single run
- Automatically creates organized directory structures based on filenames
- Uses batching for better GPU utilization

**Performance Optimization**: Our implementation uses:
- ThreadPoolExecutor with configurable workers
- Distributed processing across multiple Ollama instances
- Round-robin port assignment for load balancing
- Batch processing of sections for better resource utilization
- Subprocess-based module loading to avoid import conflicts

**Quality Improvements**:
- Enhanced regex patterns to remove thinking tags and meta-commentary
- Added Napoleon-specific relevance filtering for sections
- Created specialized prompts for military, political, and personal categories
- Implemented filtering to remove "I don't recall" responses
- Modified question generation to only ask about explicitly stated information

## Three-Stage Processing Pipeline

### Stage 1: Structural Parsing
- Extract raw text and preserve encoding
- Identify document structure (headers, sections, chapters)
- Remove modern publishing metadata
- Detect footnotes, endnotes, and editorial insertions
- Preserve document hierarchy

### Stage 2: Content Classification  
- Distinguish between voices (direct speech, quotations, narration)
- Classify content by type (description, reflection, dialogue)
- Detect stylistic patterns that indicate authorship
- Identify embedded documents (letters, speeches, journal entries)

### Stage 3: Semantic Organization & QA Generation
- Cluster content by topic across documents
- Build relationships between themes and concepts
- Generate category-specific questions about historical content
- Create first-person responses in the voice of historical figures
- Produce structured QA pairs for model fine-tuning

## Iterative Development Plan

### Phase 1: Foundation (Completed)
- ✅ Basic text ingestion and encoding detection
- ✅ Initial cleaning of obvious metadata
- ✅ Chunking for manageable processing
- ✅ Simple JSON output format

### Phase 2: Enhanced Structure Detection (Completed)
- ✅ Improved boundary detection between metadata and content
- ✅ Detection of natural section breaks within documents
- ✅ Hierarchical document processing approach
- ✅ Complete document scanning
- ✅ Multi-instance distributed processing
- ✅ Multiple document processing in a single run

### Phase 3: QA Pair Generation (Current)
- ✅ Category-specific question generation (military, political, personal)
- ✅ First-person answer generation in Napoleon's voice
- ✅ Batched processing for better resource utilization
- ✅ Automatic filtering of irrelevant sections and non-answers
- ✅ Multi-document processing with automatic directory creation
- 🔄 Quality improvement and fine-tuning for specific use cases

#### Phase 3 Implementation Details
Our QA pair generator:

1. **Document Processing Workflow**
   - Automatically runs hierarchical analysis if needed
   - Processes sections in batches for better resource utilization
   - Filters sections for Napoleon-relevance before processing
   - Organizes output in a structured directory hierarchy

2. **Specialized Question Generation**
   - Generates category-specific questions (military, political, personal)
   - Only asks about explicitly mentioned information
   - Uses second-person phrasing to address Napoleon directly
   - Deduplicates similar questions within and across categories

3. **First-Person Answer Generation**
   - Generates answers in Napoleon's authentic voice
   - Emphasizes category-appropriate attributes (military strategy, political calculation, personal emotions)
   - Filters out "I don't recall" responses for better quality
   - Adds metadata about source sections and categories

4. **Multi-Document Support**
   - Processes multiple input documents in a single run
   - Creates organized directory structure based on filenames
   - Maintains both individual and combined output files
   - Preserves document source information in QA metadata

### Phase 4: Voice and Authorship Analysis (Next)
- Train classifiers to distinguish between voices
- Detect stylistic patterns indicating authorship
- Identify first-person vs third-person narratives
- Flag content with direct speech vs biographical material

### Phase 5: Semantic Understanding
- Implement topic modeling across the corpus
- Build relationships between related content
- Create topic graphs showing conceptual relationships
- Support content retrieval by semantic similarity

## Immediate Next Steps

1. **QA generation quality improvements**
   - Fine-tune prompts for specific historical periods and figures
   - Implement better cross-category deduplication
   - Add support for contextual questions that span multiple sections
   - Create evaluation metrics for QA pair quality

2. **Performance enhancements**
   - Optimize batch sizes for different models and hardware
   - Implement parallel processing of different categories
   - Add caching for processed sections to avoid redundant computation
   - Create dynamic worker allocation based on available resources

3. **Model fine-tuning preparation**
   - Format QA pairs for specific model architectures
   - Create filtered datasets for different training objectives
   - Design evaluation prompts to test fine-tuned models
   - Build validation pipeline for generated responses

4. **User interface improvements**
   - Create a simple web interface for browsing QA pairs
   - Add visualization tools for exploring document structure
   - Implement search capabilities across generated content
   - Build a dashboard for monitoring processing status

## Key Design Decisions

1. **Hierarchical vs. Chunking**: We chose a hierarchical approach over chunking to preserve document structure and relationships.

2. **Complete Scanning**: We process the entire document rather than samples to ensure comprehensive coverage.

3. **Hybrid Detection**: We use regex for initial scanning and LLM for deeper analysis to balance efficiency and accuracy.

4. **Relationship Tracking**: We explicitly track relationships between document elements to support more sophisticated analysis.

5. **Distributed Processing**: We deploy multiple Ollama instances to better utilize system memory and increase throughput.

6. **Specialized vs. General Prompts**: We use category-specific prompts to generate higher-quality, more focused QA pairs.

## Technologies

- Python for core processing
- NLTK and spaCy for NLP tasks
- LLMs (via Ollama) for intelligent text analysis
- Machine learning for classification and clustering
- ThreadPoolExecutor for parallel processing
- Subprocess-based module loading to avoid import conflicts

## Project Structure
napoleon/
├── venv/ # Virtual environment for isolated dependencies
├── corpus/ # Raw downloaded texts
├── filtered_corpus/ # Processed and filtered texts
├── output/ # Organized output with automatically created subdirectories
├── code/ 
│ ├── hierarchical_analyzer.py # Hierarchical document analysis
│ ├── test_hierarchical.py # Script to run hierarchical analysis
│ ├── generate_qa_pairs_simple.py # Simplified QA pair generator (no llama-index)
│ ├── generate_qa_pairs.py # QA pair generator using llama-index
│ ├── structure_analyzer.py # Previous structure detection
│ ├── text_processor.py # Basic text processing
│ ├── start_ollama_instances.sh # Script to start multiple Ollama instances
│ └── napoleon_data_scraper.py # Data collection
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Usage
### Running Distributed Processing
To start multiple Ollama instances:
```bash
./code/start_ollama_instances.sh --ports 11434,11435,11436,11437
```

### Process Document Structure Hierarchically
To analyze the structure of a document using multiple Ollama instances:
```bash
python code/test_hierarchical.py --ports 11434,11435,11436,11437 path/to/file.txt
```

### Generate QA Pairs (Simplified Method)
To generate Napoleon-focused QA pairs from documents:
```bash
python code/generate_qa_pairs_simple.py path/to/file1.txt path/to/file2.txt --output_dir output --model qwen3:235b-a22b --batch_size 3
```

### Generate QA Pairs (llama-index Method)
To generate QA pairs using llama-index:
```bash
python code/generate_qa_pairs.py path/to/file.txt --output_dir output --model qwen3:235b-a22b
```

### Process Text (Original Method)
To process text files in a directory:
```bash
python code/process_texts.py input_directory output_directory
```


## Future Enhancements
- Integrate more sources such as Wikisource and The Napoleon Series
- Implement fine-tuning of models using generated QA pairs
- Build interactive visualization of document structure and QA relationships
- Create a web interface for exploring and searching generated content
- Develop a conversational AI that can respond as Napoleon using the generated data
- Optimize RAM utilization with llama.cpp and multiprocessing:
  - Replace Ollama with llama-cpp-python for direct model control
  - Use process-based parallelism to load multiple model instances in RAM
  - Disable memory-mapping (use_mmap=False, use_mlock=True) to fully load models
  - Scale processing across many workers to maximize 512GB RAM utilization
  - Implement document preloading to minimize I/O bottlenecks during processing

## License
This project is licensed under the MIT License.