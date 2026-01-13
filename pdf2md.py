"""
PDF to Markdown Converter with Ollama VLM Integration.
Converts PDF documents to Markdown while preserving layout and images.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import shutil
import re
from urllib.parse import quote

from ollama_client import OllamaClient


class PDF2Markdown:
    """Convert PDF to Markdown using VLM for layout understanding."""
    
    def __init__(self, pdf_path: str, model: str = "qwen3-vl"):
        """
        Initialize the converter.
        
        Args:
            pdf_path: Path to the PDF file
            model: Ollama vision model to use
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.doc = fitz.open(pdf_path)
        self.ollama = OllamaClient(model=model)
        self.output_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None
        
    def _setup_output_dirs(self, output_path: Optional[str] = None, images_dir: Optional[str] = None):
        """Set up output directories."""
        if output_path:
            self.output_dir = Path(output_path).parent
        else:
            self.output_dir = self.pdf_path.parent
        
        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            self.images_dir = self.output_dir / f"{self.pdf_path.stem}_images"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def _render_page_to_image(self, page: fitz.Page, dpi: int = 150) -> str:
        """
        Render a PDF page to an image file.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering
            
        Returns:
            Path to the rendered image
        """
        zoom = dpi / 72  # 72 is the default DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        # Save to temp file
        temp_path = Path(tempfile.gettempdir()) / f"pdf_page_{page.number}.png"
        pix.save(str(temp_path))
        
        return str(temp_path)
    
    def _convert_transparent_to_white(self, image_path: Path) -> None:
        """
        Convert any image to have a white background.
        This ensures images display correctly on GitHub (dark mode shows transparency as black).
        Also handles images with embedded transparency or black backgrounds.
        """
        from PIL import Image
        
        try:
            with Image.open(image_path) as img:
                # Always create a white background and composite the image onto it
                # This handles:
                # - RGBA with transparency
                # - LA (grayscale with alpha)
                # - P (palette) with transparency
                # - Images with black backgrounds that should be white
                
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                
                if img.mode in ('RGBA', 'LA'):
                    # Has alpha channel - use it as mask
                    if img.mode == 'LA':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[3])
                elif img.mode == 'P':
                    # Palette mode - may have transparency
                    if 'transparency' in img.info:
                        img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[3])
                    else:
                        img = img.convert('RGB')
                        background.paste(img)
                elif img.mode == 'RGB':
                    # No transparency - just copy
                    background.paste(img)
                elif img.mode == 'L':
                    # Grayscale - convert to RGB
                    img = img.convert('RGB')
                    background.paste(img)
                elif img.mode == 'CMYK':
                    # CMYK - convert to RGB
                    img = img.convert('RGB')
                    background.paste(img)
                else:
                    # Unknown mode - try to convert to RGBA
                    try:
                        img = img.convert('RGBA')
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[3])
                        else:
                            background.paste(img.convert('RGB'))
                    except Exception:
                        background.paste(img.convert('RGB'))
                
                # Save as PNG (overwrite original)
                background.save(image_path, 'PNG')
        except Exception as e:
            print(f"Warning: Could not process image transparency: {e}")
    
    def _extract_page_links(self, page: fitz.Page) -> List[dict]:
        """
        Extract hyperlinks from a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of link dictionaries with 'text', 'uri', and 'type' keys
        """
        links = []
        
        for link in page.get_links():
            link_info = {
                'text': '',
                'uri': '',
                'type': ''
            }
            
            # Get the link rectangle
            rect = link.get('from', fitz.Rect())
            if rect:
                # Extract text within the link's bounding box
                link_info['text'] = page.get_text("text", clip=rect).strip()
            
            # Handle different link types
            kind = link.get('kind', 0)
            
            if kind == fitz.LINK_URI:  # External URI
                link_info['uri'] = link.get('uri', '')
                link_info['type'] = 'external'
            elif kind == fitz.LINK_GOTO:  # Internal link to another page
                target_page = link.get('page', -1)
                if target_page >= 0:
                    link_info['uri'] = f'#page-{target_page + 1}'
                    link_info['type'] = 'internal'
            elif kind == fitz.LINK_NAMED:  # Named destination
                link_info['uri'] = link.get('name', '')
                link_info['type'] = 'named'
            
            if link_info['text'] and link_info['uri']:
                links.append(link_info)
        
        return links
    
    def _apply_links_to_markdown(self, markdown_content: str, links: List[dict]) -> str:
        """
        Apply extracted PDF links to markdown content.
        
        Args:
            markdown_content: The markdown text from VLM
            links: List of link dictionaries from _extract_page_links
            
        Returns:
            Markdown content with proper hyperlinks
        """
        for link in links:
            text = link['text']
            uri = link['uri']
            
            if not text or not uri:
                continue
            
            # Clean up text for matching (remove extra spaces/newlines)
            clean_text = ' '.join(text.split())
            
            # Skip if text is too short (likely false positive)
            if len(clean_text) < 3:
                continue
            
            # Escape regex special characters in the text
            escaped_text = re.escape(clean_text)
            
            # Create the markdown link
            md_link = f'[{clean_text}]({uri})'
            
            # We need to replace all occurrences that are NOT already markdown links
            # Use a function to check each match
            def replace_if_not_linked(match):
                # Get the position of the match
                start = match.start()
                end = match.end()
                
                # Check if this text is already part of a markdown link by looking for:
                # 1. Is it inside [...] of a link (preceded by [ and followed by ](?
                # 2. Is it the URL part of a link (inside parentheses after ])
                
                # Look backwards for unmatched [
                before = markdown_content[:start]
                after = markdown_content[end:]
                
                # Check if preceded by [ without a closing ]
                bracket_count = 0
                for c in reversed(before[-50:]):  # Check last 50 chars
                    if c == ']':
                        bracket_count += 1
                    elif c == '[':
                        if bracket_count > 0:
                            bracket_count -= 1
                        else:
                            # Found unmatched [ - we're inside link text
                            return match.group(0)  # Don't replace
                
                # Check if this is followed by ]( which would mean it's the end of existing link text
                if after.lstrip().startswith(']('):
                    return match.group(0)  # Don't replace
                
                return md_link
            
            # Pattern to find the text (simple word boundary matching)
            pattern = rf'(?<![[\w])({escaped_text})(?![]\w])'
            markdown_content = re.sub(pattern, replace_if_not_linked, markdown_content, flags=re.IGNORECASE)
        
        return markdown_content
    
    def _extract_page_images(self, page: fitz.Page) -> List[Tuple[str, str]]:
        """
        Extract embedded images from a page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of (image_path, description) tuples
        """
        from PIL import Image
        import io
        
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            smask = img_info[1]  # SMask xref (soft mask / alpha channel)
            
            try:
                # Extract the base image
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Create image from bytes
                pil_img = Image.open(io.BytesIO(image_bytes))
                
                # Check if there's an SMask (transparency mask)
                if smask and smask > 0:
                    try:
                        # Extract the soft mask
                        mask_image = self.doc.extract_image(smask)
                        mask_bytes = mask_image["image"]
                        mask_pil = Image.open(io.BytesIO(mask_bytes))
                        
                        # Convert mask to proper mode
                        if mask_pil.mode != 'L':
                            mask_pil = mask_pil.convert('L')
                        
                        # Ensure both images are the same size
                        if mask_pil.size != pil_img.size:
                            mask_pil = mask_pil.resize(pil_img.size, Image.LANCZOS)
                        
                        # Add alpha channel to image
                        if pil_img.mode == 'RGB':
                            pil_img = pil_img.convert('RGBA')
                        elif pil_img.mode != 'RGBA':
                            pil_img = pil_img.convert('RGBA')
                        
                        # Put the mask as alpha channel
                        pil_img.putalpha(mask_pil)
                    except Exception as e:
                        print(f"Warning: Could not apply SMask: {e}")
                
                # Save image with PDF-specific prefix for unique naming across batch processing
                pdf_prefix = self.pdf_path.stem[:20]  # Limit prefix length
                image_name = f"{pdf_prefix}_p{page.number + 1}_img{img_index + 1}.png"
                image_path = self.images_dir / image_name
                
                # Save as PNG first
                if pil_img.mode in ('RGBA', 'P'):
                    pil_img.save(image_path, 'PNG')
                else:
                    pil_img.convert('RGB').save(image_path, 'PNG')
                
                # Convert transparent background to white for GitHub compatibility
                self._convert_transparent_to_white(image_path)
                
                # Get description from VLM
                try:
                    description = self.ollama.describe_image(str(image_path))
                except Exception as e:
                    description = f"Image from page {page.number + 1}"
                    print(f"Warning: Could not describe image: {e}")
                
                images.append((str(image_path), description))
                
            except Exception as e:
                print(f"Warning: Could not extract image {img_index}: {e}")
        
        return images
    
    def convert_page(self, page_num: int) -> str:
        """
        Convert a single page to Markdown.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Markdown content for the page
        """
        if page_num >= len(self.doc):
            raise ValueError(f"Page {page_num} does not exist. PDF has {len(self.doc)} pages.")
        
        page = self.doc[page_num]
        print(f"Processing page {page_num + 1}/{len(self.doc)}...")
        
        # Extract embedded images first
        extracted_images = self._extract_page_images(page)
        
        # Extract hyperlinks from the page
        extracted_links = self._extract_page_links(page)
        
        # Render page to image for VLM analysis
        page_image_path = self._render_page_to_image(page)
        
        try:
            # Get structured content from VLM
            markdown_content = self.ollama.analyze_page_image(page_image_path)
            
            # Replace image placeholders with actual image references
            # VLM may output variations like: ![...](image_placeholder), ![...](...placeholder...), etc.
            if extracted_images:
                for img_path, description in extracted_images:
                    # Make path relative to output directory and use forward slashes for GitHub compatibility
                    rel_path = Path(img_path).relative_to(self.output_dir) if self.output_dir else Path(img_path)
                    rel_path_str = rel_path.as_posix()  # Convert to forward slashes for cross-platform
                    # URL-encode the path for GitHub compatibility (spaces, special chars)
                    # Use quote with safe='/' to preserve directory separators
                    rel_path_encoded = quote(rel_path_str, safe='/')
                    # Clean description for markdown (remove newlines, limit length)
                    clean_desc = description.replace('\n', ' ').strip()[:100]
                    img_md = f"![{clean_desc}]({rel_path_encoded})"
                    
                    # Use regex to find and replace image placeholders
                    # Match patterns like: ![any text](image_placeholder) or ![any](any_placeholder)
                    placeholder_pattern = r'!\[[^\]]*\]\([^)]*placeholder[^)]*\)'
                    if re.search(placeholder_pattern, markdown_content, re.IGNORECASE):
                        markdown_content = re.sub(
                            placeholder_pattern,
                            img_md,
                            markdown_content,
                            count=1,
                            flags=re.IGNORECASE
                        )
                    else:
                        # Append image at end if no placeholder found
                        markdown_content += f"\n\n{img_md}\n"
            
            # Clean up any remaining unfilled image placeholders
            # (VLM may reference more images than actually exist in the PDF)
            placeholder_pattern = r'!\[[^\]]*\]\([^)]*placeholder[^)]*\)'
            markdown_content = re.sub(placeholder_pattern, '', markdown_content, flags=re.IGNORECASE)
            
            # Apply extracted hyperlinks to markdown
            if extracted_links:
                markdown_content = self._apply_links_to_markdown(markdown_content, extracted_links)
            
            return markdown_content
            
        finally:
            # Cleanup temp page image
            Path(page_image_path).unlink(missing_ok=True)
    
    def convert(
        self,
        output_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Convert PDF to Markdown.
        
        Args:
            output_path: Path for output Markdown file
            images_dir: Directory to save extracted images
            page_range: Optional (start, end) tuple for page range (0-indexed, inclusive)
            
        Returns:
            Path to the generated Markdown file
        """
        self._setup_output_dirs(output_path, images_dir)
        
        if output_path:
            md_path = Path(output_path)
        else:
            md_path = self.output_dir / f"{self.pdf_path.stem}.md"
        
        # Determine page range
        start_page = 0
        end_page = len(self.doc) - 1
        
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(len(self.doc) - 1, page_range[1])
        
        print(f"Converting pages {start_page + 1} to {end_page + 1}...")
        
        # Convert pages
        all_content = []
        for page_num in range(start_page, end_page + 1):
            content = self.convert_page(page_num)
            all_content.append(content)
            all_content.append("\n\n---\n\n")  # Page separator
        
        # Remove last separator
        if all_content:
            all_content.pop()
        
        # Write output
        full_content = "".join(all_content)
        md_path.write_text(full_content, encoding="utf-8")
        
        print(f"✓ Markdown saved to: {md_path}")
        print(f"✓ Images saved to: {self.images_dir}")
        
        return str(md_path)
    
    def close(self):
        """Close the PDF document."""
        self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        with PDF2Markdown(sys.argv[1]) as converter:
            converter.convert()
    else:
        print("Usage: python pdf2md.py <pdf_file>")
