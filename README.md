# PDF to Markdown Converter

Convert PDF documents to Markdown format while preserving layout, images, and hyperlinks using Vision Language Models (VLM) through Ollama.

## Features

- **Layout Preservation**: Maintains document structure including headings, paragraphs, lists, tables, and code blocks
- **Image Extraction**: Extracts embedded images from PDFs and includes them in the Markdown output with AI-generated descriptions
- **Hyperlink Preservation**: Extracts and preserves both internal document links and external URLs
- **VLM-Powered**: Uses Ollama vision models (LLaVA, Qwen-VL, etc.) for intelligent content extraction
- **GitHub-Ready Output**: URL-encodes paths for compatibility with GitHub and other platforms

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) with a vision model installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/checko/pdf2md.git
cd pdf2md
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and a vision model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull qwen3-vl
```

## Usage

### Basic Usage

Convert an entire PDF:
```bash
python main.py document.pdf
```

### Options

```bash
python main.py <pdf_file> [options]

Options:
  --model MODEL       Ollama vision model to use (default: qwen3-vl)
  --output PATH       Output Markdown file path
  --images-dir DIR    Directory to save extracted images
  --pages START-END   Page range to convert (e.g., 1-2 for first 2 pages)
```

### Examples

```bash
# Convert entire PDF
python main.py document.pdf

# Convert first 2 pages for testing
python main.py document.pdf --pages 1-2

# Use specific model and output path
python main.py document.pdf --model llama3.2-vision --output output.md

# Specify custom images directory
python main.py document.pdf --images-dir ./my_images
```

### Remote Ollama Server

To use a remote Ollama server, set the `OLLAMA_HOST` environment variable:

```bash
# Linux/macOS
export OLLAMA_HOST="http://your-server:11434"

# Windows PowerShell
$env:OLLAMA_HOST = "http://your-server:11434"

# Then run the converter
python main.py document.pdf
```

## Output

The converter produces:
- A Markdown file (`.md`) with the document content
- An images folder (`*_images/`) containing extracted images

## Supported Vision Models

Any Ollama vision model should work, including:
- `qwen3-vl` (recommended, default)
- `llava`
- `llama3.2-vision`
- `bakllava`

## How It Works

1. **Page Rendering**: Each PDF page is rendered to an image
2. **VLM Analysis**: The vision model analyzes the page image and extracts content as Markdown
3. **Image Extraction**: Embedded images are extracted from the PDF using PyMuPDF
4. **Link Extraction**: Hyperlinks are extracted and applied to the Markdown content
5. **Post-Processing**: Image placeholders are replaced with actual image references

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
