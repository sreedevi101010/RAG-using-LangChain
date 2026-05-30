from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

#data_dir is expected to be a string 
#Python lists are heterogeneous
#here, the function returns list of any datatye.

def load_all_documents(data_dir: str) -> List[Any]:

    import os

    print("Current working directory:", os.getcwd())
    """
    
    Load all supported filesd from the data directory and convert to LangChain document structure.
    Supported file: PDF, TXT, CSV, Excel, Word, JSON
    """

    #use project root data folder

    data_path = Path(data_dir).resolve()
    #Path comes from python's pathlib module
    #.resolve() converts the relative path into absolute path, ensuring that we are working with the correct directory regardless of where the script is run from.
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    pdf_files = list(data_path.glob('**/*.pdf'))
    #glob is used to find files matching a pattern.
    # *.pdf only searches in the current folder
    # **/*.pdf searches in the current folder and all subfolders recursively for files ending with .pdf
    # glob returns a iterator, so we convert it to a list to easily count and print the found files.
    # if list is not used, only len won't work
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    #len(pdf_files) counts the number of PDF files found, and the list comprehension [str(f) for f in pdf_files] creates a list of the file paths as strings for easier debugging output.
    """
    before:
    pdf_files = [
        PosixPath('/data/report.pdf'),
        PosixPath('/data/ml.pdf')
    ]

    after:
    [
        '/data/report.pdf',
        '/data/ml.pdf'
    ]
    """
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF file: {pdf_file}")
        # try doing so and if there is an exception, catch it and print the error message.
        #This is important because one corrupted PDF shouldn't stop loading all the others.
        try:
            #pdf_file is a Path object, and PyPDFLoader expects a string path, so we convert it using str(pdf_file).
            loader=PyPDFLoader(str(pdf_file))
            loaded=loader.load()
            #load() is where langchain reads the pdf. It returns a list of documents, where each document corresponds to a page in the PDF. Each document is a dictionary containing the page content and metadata.
            """
            loaded is printed below
            [
                Document(page_content="Page 1 text", metadata={...}),
                Document(page_content="Page 2 text", metadata={...}),
                Document(page_content="Page 3 text", metadata={...}),
                Document(page_content="Page 4 text", metadata={...}),
                Document(page_content="Page 5 text", metadata={...})
            ]
            
            """
            print(f"[DEBUG] Loaded {len(loaded)} PDF pages from {pdf_file}")
            documents.extend(loaded)
            """
            if append is used,
            [
                [page1, page2, page3],
                [page4, page5]
            ]
            ie, a nested list.
            """
        except Exception as e:
            print(f"[ERROR] Failed to load PDF file: {pdf_file}, Error: {e}")
            # reasons - no permission, file not found, corrupted file, unsupported format, proteced/encrypted file, out of memory etc.





    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")





     # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")




    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")




    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")




    
    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")



    return documents


if __name__ == "__main__":
    docs=load_all_documents("data")
    # the path given should be relative to current working directory , not this
    print(f"loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)