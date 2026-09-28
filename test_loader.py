from app.document_loader import load_all_pdfs


documents = load_all_pdfs("data")


print(f"Total pages loaded: {len(documents)}")

for document in documents:

    print("\n-----------------------------")
    print("Source:", document["source"])
    print("Page:", document["page"])
    print("Text:")
    print(document["text"])