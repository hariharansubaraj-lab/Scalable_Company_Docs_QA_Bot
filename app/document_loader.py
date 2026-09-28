from pypdf import PdfReader


def load_pdf(file, filename):

    reader = PdfReader(file)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            # Clean unnecessary spaces
            lines = []

            for line in text.splitlines():

                line = " ".join(line.split())

                if line:
                    lines.append(line)

            clean_text = "\n".join(lines)

            if clean_text.strip():

                documents.append({
                    "text": clean_text.strip(),
                    "source": filename,
                    "page": page_number
                })

    return documents