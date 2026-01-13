"""
PDF to Markdown Converter CLI.

Usage:
    python main.py <pdf_file> [options]
    python main.py --folder <directory> [options]
    
Options:
    --model MODEL       Ollama vision model (default: qwen3-vl)
    --output PATH       Output Markdown file path (single file only)
    --images-dir DIR    Directory to save extracted images
    --pages START-END   Page range to convert (e.g., 1-2 for first 2 pages)
    --folder DIR        Convert all PDF files in the specified directory
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


def convert_single_pdf(pdf_path: Path, args):
    """Convert a single PDF file to Markdown."""
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
    return output_path


def convert_folder(folder_path: Path, args):
    """Convert all PDF files in a folder to Markdown."""
    pdf_files = list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.PDF"))
    
    if not pdf_files:
        print(f"No PDF files found in: {folder_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"=== Batch PDF to Markdown Converter ===")
    print(f"Folder: {folder_path}")
    print(f"Model:  {args.model}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    if args.pages:
        print(f"Pages:  {args.pages[0] + 1} to {args.pages[1] + 1}")
    print()
    
    successful = []
    failed = []
    
    for i, pdf_file in enumerate(sorted(pdf_files), 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print(f"{'='*50}")
        
        try:
            with PDF2Markdown(str(pdf_file), model=args.model) as converter:
                output_path = converter.convert(
                    images_dir=args.images_dir,
                    page_range=args.pages
                )
            successful.append((pdf_file.name, output_path))
            print(f"✓ Completed: {pdf_file.name}")
        except Exception as e:
            failed.append((pdf_file.name, str(e)))
            print(f"✗ Failed: {pdf_file.name} - {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"=== Batch Conversion Summary ===")
    print(f"{'='*50}")
    print(f"Total:      {len(pdf_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    
    if successful:
        print(f"\nConverted files:")
        for name, output in successful:
            print(f"  ✓ {name} -> {output}")
    
    if failed:
        print(f"\nFailed files:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")
    
    return len(failed) == 0


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
    
    # Convert all PDFs in a folder
    python main.py --folder /path/to/pdfs
    
    # Convert all PDFs in a folder with page range
    python main.py --folder /path/to/pdfs --pages 1-5
"""
    )
    
    parser.add_argument("pdf_file", nargs="?", help="Path to the PDF file to convert")
    parser.add_argument("--folder", "-f",
                        help="Convert all PDF files in the specified directory")
    parser.add_argument("--model", default="qwen3-vl", 
                        help="Ollama vision model to use (default: qwen3-vl)")
    parser.add_argument("--output", "-o", 
                        help="Output Markdown file path (only for single file mode)")
    parser.add_argument("--images-dir", 
                        help="Directory to save extracted images")
    parser.add_argument("--pages", type=parse_page_range,
                        help="Page range to convert (e.g., '1-2' for pages 1-2)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.folder and args.pdf_file:
        print("Error: Cannot specify both pdf_file and --folder. Use one or the other.", 
              file=sys.stderr)
        sys.exit(1)
    
    if not args.folder and not args.pdf_file:
        parser.print_help()
        sys.exit(1)
    
    if args.folder and args.output:
        print("Warning: --output is ignored in folder mode. Each PDF will be saved with its original name.", 
              file=sys.stderr)
    
    try:
        if args.folder:
            # Folder mode: convert all PDFs in the directory
            folder_path = Path(args.folder)
            if not folder_path.exists():
                print(f"Error: Folder not found: {args.folder}", file=sys.stderr)
                sys.exit(1)
            if not folder_path.is_dir():
                print(f"Error: Not a directory: {args.folder}", file=sys.stderr)
                sys.exit(1)
            
            success = convert_folder(folder_path, args)
            sys.exit(0 if success else 1)
        else:
            # Single file mode
            pdf_path = Path(args.pdf_file)
            if not pdf_path.exists():
                print(f"Error: PDF file not found: {args.pdf_file}", file=sys.stderr)
                sys.exit(1)
            
            convert_single_pdf(pdf_path, args)
        
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
