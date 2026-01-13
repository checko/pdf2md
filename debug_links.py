"""Debug script to check PDF links extraction."""
import fitz

pdf_path = r'2.2.3.1. Android Development — NeuroPilot 8.0.9 basic documentation.pdf'
doc = fitz.open(pdf_path)

print(f"PDF has {len(doc)} pages\n")

# Check last page (page 16, index 15)
page = doc[15]
links = page.get_links()

print(f"=== Page 16 (last page) ===")
print(f"Found {len(links)} links:\n")

for i, link in enumerate(links):
    rect = link.get('from', fitz.Rect())
    if rect:
        text = page.get_text('text', clip=rect).strip()
    else:
        text = "(no rect)"
    
    kind = link.get('kind', 0)
    kind_name = {0: 'NONE', 1: 'GOTO', 2: 'URI', 3: 'LAUNCH', 4: 'NAMED', 5: 'GOTOR'}.get(kind, str(kind))
    uri = link.get('uri', '')
    target_page = link.get('page', -1)
    
    print(f"Link {i+1}:")
    print(f"  Kind: {kind_name} ({kind})")
    print(f"  Text: \"{text}\"")
    print(f"  URI: {uri}")
    print(f"  Target Page: {target_page}")
    print()

doc.close()
