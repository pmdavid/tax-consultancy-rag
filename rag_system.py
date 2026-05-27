from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st
from dotenv import load_dotenv
from config import *
from prompts import *
load_dotenv()

# INIT RAG ARCHITECTURE
@st.cache_resource # This is a decorator provided by Streamlit that allows us to cache the result of the function, so that it only runs once and then stores the result in cache. This is particularly useful for initializing the RAG system, which can be resource-intensive, as it avoids re-running the initialization every time a query is made or the interface is updated. By caching the initialized RAG system, we can significantly improve the performance and responsiveness of our application, especially when dealing with large vector stores and complex retriever configurations.
def initialize_rag_system():

    # Vector Store
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    # Models
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0) # modelo más rápido para generar consultas
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0) # modelo más potente para generar respuestas

    # Retriever MMR (Maximal Margin Relevance)
    base_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Additional retriever, similarity, useful to compare
    # Pros: Semantic understanding and simplicity
    # Cons: The problem with diversity: If you ask for $k=4$, it may sometimes return 4 fragments that say exactly the same thing using slightly different wording. This wastes space in the LLM's context window.
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever with custom prompt. It generates multiple queries from the original question to retrieve a more diverse set of relevant documents, which can help improve the quality of the generated response by providing a richer context. The custom prompt allows us to guide the query generation process, ensuring that the generated queries are focused and relevant to the user's original question, which can lead to better retrieval results and ultimately a more accurate and informative response from the LLM.
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # Ensemble Retriever que combinar MMR y similarity (similitud de coseno) para mejorar la relevancia de los documentos recuperados, dando más peso a MMR pero también considerando la similitud directa.
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # Distribuimos peso de importancia entre los dos retrievers
            similarity_threshold=SIMILARITY_THRESHOLD # Solo se considerarán documentos del similarity_retriever si superan este umbral de similitud, lo que ayuda a filtrar resultados menos relevantes y mantener la calidad de los documentos recuperados.
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formatear y preprocesar los documentos/fragmentos recuperados desde el retriever,
    # añadiendo encabezados y metadatos relevantes para mejorar la generación de respuestas.
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"
        
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)


    rag_chain = (
        {   # Los resultados devueltos del retriever se pasan a través de la función de formateo y eso será el "contexto" que se le da al prompt para generar la respuesta.
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever

# Función para manejar la consulta del usuario, obtener la respuesta generada por el modelo y los documentos relevantes recuperados, formateándolos para su presentación en la interfaz.
def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta, que viene del rag_chain montado anteriormente
        response = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)
        
        return response, docs_info
    
    except Exception as e:
        error_msg = f"Error al procesar la cosulta: {str(e)}"
        return error_msg, []

# Funcion auxiliar para obtener información sobre la configuración del retriever, que se puede mostrar en la interfaz para que el usuario entienda cómo se están recuperando los documentos relevantes.
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }