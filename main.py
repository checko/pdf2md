"""
PDF to Markdown Converter CLI.

Usage:
    python main.py <pdf_file> [options]
    
Options:
    --model MODEL       Ollama vision model (default: llava)
    --output PATH       Output Markdown file path
    --images-dir DIR    Directory to save extracted images
    --pages START-END   Page range to convert (e.g., 1-2 for first 2 pages)
"""

import argparse
import sys
from pathlib import Path

from pdf2md import PDF2Markdown


def parse_page_range(page_str: str) -> tuple:
    """Parse page range string like '1-3' to (0, 2) tuple."""
    try:
        if '-' in page_str:
            start, end = page_str.split('-')
            return (int(start) - 1, int(end) - 1)  # Convert to 0-indexed
        else:
            page = int(page_str) - 1
            return (page, page)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid page range: {page_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown with layout preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert entire PDF
    python main.py document.pdf
    
    # Convert first 2 pages for testing
    python main.py document.pdf --pages 1-2
    
    # Use specific model and output path
    python main.py document.pdf --model llama3.2-vision --output output.md
"""
    )
    
    parser.add_argument("pdf_file", help="Path to the PDF file to convert")
    parser.add_argument("--model", default="llava", 
                        help="Ollama vision model to use (default: llava)")
    parser.add_argument("--output", "-o", 
                        help="Output Markdown file path")
    parser.add_argument("--images-dir", 
                        help="Directory to save extracted images")
    parser.add_argument("--pages", type=parse_page_range,
                        help="Page range to convert (e.g., '1-2' for pages 1-2)")
    
    args = parser.parse_args()
    
    # Validate PDF exists
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"=== PDF to Markdown Converter ===")
        print(f"Input:  {pdf_path}")
        print(f"Model:  {args.model}")
        if args.pages:
            print(f"Pages:  {args.pages[0] + 1} to {args.pages[1] + 1}")
        print()
        
        with PDF2Markdown(str(pdf_path), model=args.model) as converter:
            output_path = converter.convert(
                output_path=args.output,
                images_dir=args.images_dir,
                page_range=args.pages
            )
        
        print(f"\n=== Conversion Complete ===")
        print(f"Output: {output_path}")
        
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
