import logging
import chromadb
from src.config import Config
from src.embeddings.embedder import Embedder

# Robust imports for multi-version LlamaIndex compatibility
try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except ImportError:
    from llama_index.vector_stores import ChromaVectorStore

try:
    from llama_index.core import Settings

    HAS_SETTINGS = True
except ImportError:
    try:
        from llama_index.settings import Settings

        HAS_SETTINGS = True
    except ImportError:
        HAS_SETTINGS = False

try:
    from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext
except ImportError:
    from llama_index import StorageContext, VectorStoreIndex, ServiceContext

logger = logging.getLogger(__name__)


class Retriever:
    """Manages index initialization, loading, inserting, and retrieval operations on ChromaDB"""

    def __init__(self, config: Config):
        self.config = config
        # Initialize persistent client
        self.db = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        self.chroma_collection = self.db.get_or_create_collection("rag_pipeline")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.embedder = Embedder(config)

        # Build ServiceContext or configure Settings using our custom embedder
        if HAS_SETTINGS:
            logger.info("Configuring global Settings (LlamaIndex 0.10+)...")
            Settings.embed_model = self.embedder.get_embedding_model()
            Settings.llm = None  # Retrieval and chunking only requires embedding model
            self.service_context = None
        else:
            logger.info("Configuring ServiceContext (LlamaIndex 0.9.x)...")
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embedder.get_embedding_model(),
                llm=None,  # Retrieval and chunking only requires embedding model
            )

    def build_index(self, chunks):
        """Build VectorStoreIndex from chunks (nodes)"""
        try:
            logger.info(f"Building index with {len(chunks)} chunks...")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            kwargs = {}
            if not HAS_SETTINGS:
                kwargs["service_context"] = self.service_context

            index = VectorStoreIndex(
                nodes=chunks, storage_context=storage_context, **kwargs
            )
            return index
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise e

    def load_index(self):
        """Load existing index from Chroma vector store"""
        try:
            logger.info("Attempting to load index from Chroma...")
            count = self.chroma_collection.count()
            if count == 0:
                logger.info("Chroma database collection is empty. No index loaded.")
                return None

            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            kwargs = {}
            if not HAS_SETTINGS:
                kwargs["service_context"] = self.service_context

            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, **kwargs
            )
            logger.info(f"Successfully loaded index with {count} items.")
            return index
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            return None

    def add_to_index(self, index, chunks):
        """Add new chunks/nodes to an existing index"""
        try:
            if not index:
                logger.error("Cannot insert to null index.")
                return
            logger.info(f"Adding {len(chunks)} new chunks to existing index...")
            index.insert_nodes(chunks)
            logger.info("Successfully added chunks to index.")
        except Exception as e:
            logger.error(f"Error inserting chunks to index: {e}")
            raise e

    def retrieve(self, index, query: str, top_k: int = 5, threshold: float = 0.5):
        """Retrieve relevant nodes matching the query, filtered by similarity threshold"""
        try:
            logger.info(
                f"Querying index (top_k={top_k}, threshold={threshold}): '{query}'"
            )
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes_with_scores = retriever.retrieve(query)

            filtered_nodes = []
            for nws in nodes_with_scores:
                score = nws.score if nws.score is not None else 1.0
                logger.info(
                    f"Retrieved node (score={score:.4f}): '{nws.node.text[:60]}...'"
                )

                # Check similarity threshold (we filter if score >= threshold)
                if score >= threshold:
                    filtered_nodes.append(nws.node)

            # Safety fallback: if everything is filtered out, return the single best match
            if not filtered_nodes and nodes_with_scores:
                logger.info(
                    "No nodes exceeded the threshold. Returning top match as fallback."
                )
                filtered_nodes.append(nodes_with_scores[0].node)

            return filtered_nodes
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []
