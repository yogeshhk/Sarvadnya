from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

# New test for converting a PDF file from a local path
source = "data/LinkedInProfile.pdf"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
