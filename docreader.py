from docling.document_converter import DocumentConverter
import sys
import os

def read_document(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return None
            
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"Error reading document: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python docreader.py <pdf_file>")
        print("Example: python docreader.py case1.pdf")
        return
    
    pdf_file = sys.argv[1]
    
    print(f"Reading PDF file: {pdf_file}")
    content = read_document(pdf_file)
    
    if content:
        print("Document content extracted successfully!")
        print("=" * 50)
        print(content[:1000] + "..." if len(content) > 1000 else content)
    else:
        print("Failed to extract document content")

if __name__ == "__main__":
    main()