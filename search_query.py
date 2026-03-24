from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from PIL import Image
import os


CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the context below.
If images are provided, use them to enhance your answer.

Context:
{context}

Question: {query}

Answer:
""")


def search_query(query, model, embedding_model):
    retriever = load_retriever(embedding_model)
    text_chunks, image_chunks = retrieve(query, retriever)

    content = build_prompt(query, text_chunks, image_chunks)
    response = model.invoke([HumanMessage(content=content)])
    answer = response.content
    images= show_images(image_chunks, model, query)

    return answer, images


def build_prompt(query, text_chunks, image_chunks):
    context = "\n\n".join([doc.page_content for doc in text_chunks])
    if image_chunks:
        image_summaries = "\n\n".join([
            f"[Image {i+1}]: {doc.page_content}"
            for i, doc in enumerate(image_chunks)
        ])
        context += f"\n\nImage Descriptions:\n{image_summaries}"

    formatted = prompt_template.format_messages(context=context, query=query)
    text_content = formatted[0].content

    content = [{"type": "text", "text": text_content}]

    return content


def show_images(image_chunks, model, query):
    images=[]
    for i, doc in enumerate(image_chunks):
        summary = doc.page_content
        relevance = model.invoke(
            f"Query: {query}\n"
            f"Image Summary: {summary}\n\n"
            f"Is this image relevant to the query? Answer only YES or NO."
        )

        if "yes" not in relevance.content.lower():
            continue

        image_path = doc.metadata.get("image_path")
        if image_path and os.path.exists(image_path):
            images.append(image_path)
        else:
            print("Image file not found:", image_path)
    return images

def load_retriever(embedding_model):
    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    all_docs = vectorstore.similarity_search("", k=10000)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever


def retrieve(query: str, retriever):
    results = retriever.invoke(query)

    text_chunks = []
    image_chunks = []

    for doc in results:
        if doc.metadata.get("type") == "image_summary":
            image_chunks.append(doc)
        else:
            text_chunks.append(doc)

    return text_chunks, image_chunks


if __name__ == "__main__":
    load_dotenv()
    query = "Image is O. Henry"
    model = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    answer, images = search_query(query, model, embeddings)
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")
    for image_path in images:
        Image.open(image_path).show()
