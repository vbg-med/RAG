import time
import logging
import threading
from pathlib import Path
from src.config import Config

logger = logging.getLogger(__name__)


class DocumentWatcher:
    """Watches the data/incoming directory for new PDF files and indexes them automatically in the background"""

    def __init__(
        self,
        config: Config,
        pdf_loader,
        image_extractor,
        chunker,
        retriever,
        get_index_func,
        set_index_func,
    ):
        self.config = config
        self.pdf_loader = pdf_loader
        self.image_extractor = image_extractor
        self.chunker = chunker
        self.retriever = retriever
        self.get_index = get_index_func
        self.set_index = set_index_func

        self.processed_files = set()
        self._running = False
        self._thread = None

    def start(self):
        """Start the background watcher thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Background document watcher thread started.")

    def stop(self):
        """Stop the background watcher thread"""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Background document watcher thread stopped.")

    def _watch_loop(self):
        # Allow system to start up fully before first scan
        time.sleep(5)
        while self._running:
            try:
                self._check_for_new_documents()
            except Exception as e:
                logger.error(f"Error in document watcher loop: {e}")
            # Poll every 30 seconds
            time.sleep(30)

    def _check_for_new_documents(self):
        incoming_dir = self.config.DATA_DIR / "incoming"
        if not incoming_dir.exists():
            return

        pdf_files = list(incoming_dir.glob("*.pdf"))

        for pdf_path in pdf_files:
            file_stat = pdf_path.stat()
            file_key = f"{pdf_path.name}_{file_stat.st_mtime}_{file_stat.st_size}"

            if file_key not in self.processed_files:
                logger.info(f"Watcher detected new/modified document: {pdf_path.name}")

                # Check that the file is not currently being copied (size stable for 1s)
                size1 = pdf_path.stat().st_size
                time.sleep(1)
                size2 = pdf_path.stat().st_size
                if size1 != size2:
                    logger.info(
                        f"File {pdf_path.name} is still writing. Skipping index addition for now."
                    )
                    continue

                self._process_and_index(pdf_path)
                self.processed_files.add(file_key)

    def _process_and_index(self, pdf_path: Path):
        try:
            logger.info(f"Processing background auto-indexing for: {pdf_path.name}")

            # Load text
            text_docs = self.pdf_loader.load(str(pdf_path))
            documents = list(text_docs)

            # Extract and caption images
            if self.config.USE_IMAGE_CAPTIONS:
                image_docs = self.image_extractor.process_pdf(str(pdf_path))
                documents.extend(image_docs)

            if not documents:
                logger.warning(
                    f"No contents parsed from {pdf_path.name}. Skipping indexing."
                )
                return

            # Split into chunks
            chunks = self.chunker.chunk_documents(documents)
            if not chunks:
                logger.warning("No chunks generated. Skipping indexing.")
                return

            # Check current global index state and update
            index = self.get_index()
            if index:
                self.retriever.add_to_index(index, chunks)
                logger.info(
                    f"Added {len(chunks)} chunks from {pdf_path.name} to existing active index."
                )
            else:
                logger.info(
                    f"Building initial index with {len(chunks)} chunks from {pdf_path.name}..."
                )
                new_index = self.retriever.build_index(chunks)
                self.set_index(new_index)
                logger.info("Initial index built successfully in background.")

        except Exception as e:
            logger.error(f"Failed to background index {pdf_path.name}: {e}")
