"""
Ollama VLM Client for PDF analysis and image description.
"""

import base64
import os
import ollama
from pathlib import Path

# Ollama server configuration - defaults to localhost, can be overridden via environment variable
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaClient:
    """Client for interacting with Ollama vision models."""
    
    def __init__(self, model: str = "qwen3-vl", host: str = OLLAMA_HOST):
        """
        Initialize the Ollama client.
        
        Args:
            model: The vision model to use (e.g., 'qwen3-vl', 'llava', 'llama3.2-vision')
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            models_response = self.client.list()
            # Handle both Pydantic model and dict response formats
            if hasattr(models_response, 'models'):
                available = [m.model for m in models_response.models]
            else:
                available = [m['name'] for m in models_response.get('models', [])]
            
            available_base = [m.split(':')[0] for m in available]
            
            print(f"Connected to Ollama at {self.host}")
            print(f"Available models: {available}")
            
            if self.model.split(':')[0] not in available_base:
                print(f"Warning: Model '{self.model}' not found.")
                # Try to find a vision model
                vision_keywords = ['llava', 'vision', 'vl', 'qwen3-vl']
                vision_models = [m for m in available if any(v in m.lower() for v in vision_keywords)]
                if vision_models:
                    self.model = vision_models[0]
                    print(f"Using available vision model: {self.model}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.host}. Error: {e}")
    
    def _clean_vlm_output(self, content: str) -> str:
        """
        Clean VLM output by removing thinking noise and markdown wrappers.
        
        Some VLMs include their internal reasoning in the output like:
        'Wait, no...', 'Let me think...', 'So final Markdown:', etc.
        This method removes such noise.
        """
        import re
        
        # Remove thinking/reasoning patterns
        thinking_patterns = [
            r'\n*Wait,\s*no[^\n]*\n',  # "Wait, no..."
            r'\n*Let me[^\n]*\n',  # "Let me think/check..."
            r'\n*So final [Mm]arkdown:[^\n]*\n',  # "So final Markdown:"
            r'\n*Actually,[^\n]*\n',  # "Actually,..."
            r'\n*I notice[^\n]*\n',  # "I notice..."
            r'\n*Looking at[^\n]*\n',  # "Looking at the image..."
        ]
        
        for pattern in thinking_patterns:
            content = re.sub(pattern, '\n', content, flags=re.IGNORECASE)
        
        # Remove markdown code block wrapper if present (```markdown ... ```)
        content = re.sub(r'^```(?:markdown)?\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        
        # Clean up multiple consecutive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def analyze_page_image(self, image_path: str) -> str:
        """
        Analyze a PDF page image to extract structured content as Markdown.
        
        Args:
            image_path: Path to the page image
            
        Returns:
            Markdown formatted content of the page
        """
        prompt = """Analyze this PDF page image and convert its content to Markdown format.

CRITICAL: Pay special attention to CODE BLOCKS and SHELL COMMANDS that may appear in LIGHT GRAY BOXES 
with GRAY or FAINT TEXT. These low-contrast code sections are VERY IMPORTANT and MUST NOT be skipped.
Look carefully for command-line examples with $ prompts, even if the text appears light or faded.

Rules:
1. Preserve the document structure (headings, paragraphs, lists, tables, code blocks)
2. CAREFULLY EXTRACT ALL CODE BLOCKS - especially those in gray/light boxes with faint text:
   - Shell commands starting with $ (e.g., $ bash script.sh, $ ./compile.sh, $ adb push)
   - File paths and command examples, even if they appear in low contrast
   - Do NOT skip any code section, even if the text is light gray on light background
3. Use proper Markdown syntax:
   - # for main titles, ## for sections, ### for subsections
   - - or * for bullet lists, 1. 2. 3. for numbered lists
   - | for tables with proper alignment
   - ``` for code blocks (use ```bash for shell commands)
   - > for quotes or callouts/notes
4. For images/diagrams in the page, write: ![Description of the image](image_placeholder)
5. Maintain reading order (top to bottom, left to right)
6. Keep text accurate - do not paraphrase or summarize
7. For headers/footers, you can skip or mark as <!-- header --> or <!-- footer -->

Output ONLY the Markdown content, no explanations."""

        response = self.client.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        
        # Clean up VLM output to remove thinking noise
        content = response['message']['content']
        return self._clean_vlm_output(content)
    
    def describe_image(self, image_path: str) -> str:
        """
        Generate a description for an extracted image/diagram.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Text description of the image
        """
        prompt = """Describe this image/diagram concisely for a Markdown document.
Focus on:
- What type of image it is (screenshot, diagram, chart, illustration)
- Key elements and their relationships
- Any text visible in the image
- The purpose or meaning it conveys

Provide a 1-3 sentence description suitable for an image alt-text."""

        response = self.client.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        
        return response['message']['content'].strip()


if __name__ == "__main__":
    # Quick test
    client = OllamaClient()
    print(f"Ollama client initialized with model: {client.model}")
