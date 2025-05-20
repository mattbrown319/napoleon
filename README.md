# Napoleon Corpus Collection

This project is dedicated to collecting, processing, and analyzing the writings of Napoleon Bonaparte. The goal is to build a training dataset for fine-tuning a language model to "talk, act, and think" like Napoleon.

## Project Structure
napoleon/
├── venv/ # Virtual environment for isolated dependencies
├── corpus/ # Raw downloaded texts
├── segmented_corpus/ # Processed text segments
├── code/ # Python scripts
│ ├── inspect_project_gutenberg.py
│ └── napoleon_data_scraper.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Installation
1. Clone this repository:
git clone https://github.com/yourusername/napoleon.git
cd napoleon

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install required packages:
pip install -r requirements.txt


## Usage
### Inspect Project Gutenberg Structure
To verify the structure of the Gutenberg page:
python code/inspect_project_gutenberg.py


### Scrape Napoleon's Writings
To collect texts from Project Gutenberg:
python code/napoleon_data_scraper.py


## Future Enhancements
- Integrate more sources such as Wikisource and The Napoleon Series.
- Automate text cleaning and context extraction.

## License
This project is licensed under the MIT License.