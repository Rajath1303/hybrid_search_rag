from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import base64
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_postgres import PGVector

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def ingest(filepath: str, model, embedding_model):
    docs = doc_loader(filepath)
    chunks = chunk_docs(docs, model)
    store_chunks(chunks, embedding_model)
    return chunks

def chunk_docs(docs, model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 50
    )
    chunks = []
    for doc in docs:
        category = doc.metadata.get("category")
        if category == "Image":
            image_path = doc.metadata.get("image_path")
            if image_path and os.path.exists(image_path):
                summary = summarize_image(image_path, model)
                chunks.append(Document(
                    page_content=summary,
                    metadata={
                        **doc.metadata,
                        "image_path": image_path,   # ← pointer to original image
                        "type": "image_summary"
                    },
                    languages=["eng"]
                ))
            else:
                chunks.append(doc)
        elif category == "Table":
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))
    print("Length of chunks after chunking: ", len(chunks))
    return chunks

def summarize_image(image_path, model):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    response = model.invoke([{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            },
            {
                "type": "text",
                "text": "Describe this image in detail. If it contains a chart or diagram, explain what it shows."
            }
        ]
    }])
    return response.content

def store_chunks(chunks, embedding_model):
    print("Ingesting into PGVector...")

    PGVector.from_documents(
        documents=chunks,
        embedding=embedding_model,
        connection=CONNECTION_STRING,  
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,
        use_jsonb=True
    )

    print("Ingestion complete")

def doc_loader(filepath: str):
    loader = UnstructuredPDFLoader(
        file_path=filepath, 
        mode="elements",
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir="extracted_images/"
    )
    docs = loader.load()
    return docs
def test_ingestion(embedding_model):

    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    results = vectorstore.similarity_search("a", k=100)
    print("Total chunks in DB:", len(results))
    query = "Image of O.Henry"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nTop 3 results for '{query}':")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print("Category :", doc.metadata.get("category"))
        print("Page     :", doc.metadata.get("page_number"))
        print("Content  :", doc.page_content[:200])
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    print("\nResults with similarity scores:")
    for doc, score in results_with_scores:
        print(f"Score: {score:.4f} | Content: {doc.page_content[:100]}")

    all_results = vectorstore.similarity_search("image diagram", k=10)
    image_chunks = [r for r in all_results if r.metadata.get("type") == "image_summary"]
    print(f"\nImage chunks found: {len(image_chunks)}")
    for chunk in image_chunks:
        print("Image path:", chunk.metadata.get("image_path"))
        print("Summary   :", chunk.page_content[:200])

    table_chunks = [r for r in all_results if r.metadata.get("category") == "Table"]
    print(f"\nTable chunks found: {len(table_chunks)}")

if __name__ == "__main__":
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o")
    embedding_model = OpenAIEmbeddings()
    ingest(filepath="docs/book4.pdf", model= model, embedding_model=embedding_model)
    test_ingestion(embedding_model)