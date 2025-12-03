"Document management implementation."

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.logging_config import log_user_action, log_sensitive_debug
from app.modules.core.version_manager import VersionManager
from app.modules.vector_db.chroma_impl import ChromaVectorStore
from app.modules.vector_db.embedding_manager import EmbeddingManager
from app.modules.core.utils import (
    chunk_text_basic,
    sanitize_metadata_dict,
    is_collection_empty,
)
from app.modules.config.settings import settings
from app.utils.doc_parser import parse_file

logger = logging.getLogger(__name__)


class DocumentManager:
    """Document management implementation."""

    def __init__(self, vector_store: ChromaVectorStore, version_manager: VersionManager, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.version_manager = version_manager
        self.embedding_manager = embedding_manager

    @staticmethod
    def _generate_ids(prefix: str, n: int) -> List[str]:
        """Generate a list of unique IDs with a given prefix."""
        return [f"{prefix}_{uuid.uuid4().hex}" for _ in range(n)]

    @staticmethod
    def _generate_document_id(source_name: str) -> str:
        """Generate a stable document ID from source name."""
        # Use hash of source name for deterministic document_id
        hash_obj = hashlib.md5(source_name.encode())
        return f"doc_{hash_obj.hexdigest()[:16]}"

    def _calculate_next_version(self, document_id: str) -> str:
        """Calculate the next version number for a document."""
        return self.version_manager.generate_next_version(document_id)

    async def add_document_to_rag_local(
            self,
            source_name: str,
            text: str,
            chunks: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            document_id: Optional[str] = None,
            version: Optional[str] = None,
            parent_version: Optional[str] = None,
            status: str = "published",
            version_notes: Optional[str] = None,
            created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a document (or precomputed chunks) to the local chroma collection with versioning support.
        """
        # Generate document_id if not provided
        if not document_id:
            document_id = self._generate_document_id(source_name)

        # Calculate version if not provided
        if not version:
            version = self._calculate_next_version(document_id)

        if not chunks:
            chunks = chunk_text_basic(text)

        if not chunks:
            logger.warning("No chunks produced for document: %s", source_name)
            return {
                "ids": [],
                "document_id": document_id,
                "version": version,
                "chunk_count": 0
            }

        # sanitize metadata and ensure source is present
        base_meta = metadata or {}
        sanitized_base = sanitize_metadata_dict(base_meta)
        sanitized_base["source"] = source_name

        # Add version metadata
        sanitized_base["document_id"] = document_id
        sanitized_base["version"] = version
        sanitized_base["version_created_at"] = datetime.utcnow().isoformat() + "Z"
        sanitized_base["version_created_by"] = created_by
        sanitized_base["parent_version"] = parent_version
        sanitized_base["status"] = status
        sanitized_base["is_latest_version"] = True  # Will be updated if newer version created

        logger.debug("Ingest metadata keys (sample): %s", list(sanitized_base.keys())[:8])

        # add ingestion timestamp if not present
        if "ingested_at" not in sanitized_base:
            sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

        metadatas = [dict(sanitized_base) for _ in chunks]
        ids = self._generate_ids(prefix=f"{document_id}_v{version}", n=len(chunks))

        log_user_action(
            logger, "DOCUMENT_INGESTION_START", created_by,
            source_name=source_name, document_id=document_id, version=version,
            chunk_count=len(chunks), status=status, has_parent=bool(parent_version)
        )

        log_sensitive_debug(
            logger, "Document ingestion details",
            ids_sample=ids[:3], metadata_keys=list(sanitized_base.keys()),
            chunk_lengths=[len(c) for c in chunks[:3]]
        )

        # compute embeddings locally
        try:
            embeddings = await self.embedding_manager.encode(chunks)
        except Exception as e:
            logger.exception("Failed to compute embeddings locally: %s", e)
            raise

        # Add to chroma via helper
        try:

            self.vector_store.add_documents_to_collection(documents=chunks, metadatas=metadatas, ids=ids,
                                                          embeddings=embeddings)

            log_user_action(
                logger, "DOCUMENT_INGESTION_SUCCESS", created_by,
                source_name=source_name, document_id=document_id, version=version,
                chunk_count=len(chunks), collection=settings.DEFAULT_COLLECTION_NAME
            )
        except Exception as e:
            logger.exception("Failed to add documents to Chroma collection: %s", e)
            raise

        # Create version record in version tracking database
        try:
            await self.version_manager.create_version_record(
                document_id=document_id,
                version=version,
                source_name=source_name,
                chunk_ids=ids,
                created_by=created_by,
                parent_version=parent_version,
                status=status,
                version_notes=version_notes,
                metadata=sanitized_base
            )
            logger.info("Created version record: document_id=%s version=%s", document_id, version)
        except Exception as e:
            logger.warning("Failed to create version record (non-fatal): %s", e)

        # Mark previous version as not latest (if this is an update)
        if parent_version:
            try:
                # Update previous version's is_latest_version flag
                prev_version_info = self.version_manager.get_version(document_id, parent_version)
                if prev_version_info:
                    prev_chunk_ids = prev_version_info["chunk_ids"]
                    self.vector_store.update_metadatas(ids=prev_chunk_ids,
                                                       metadata={"is_latest_version": False})
                    logger.info("Marked previous version %s as not latest", parent_version)
            except Exception as e:
                logger.warning("Failed to update previous version metadata (non-fatal): %s", e)

        return {
            "ids": ids,
            "document_id": document_id,
            "version": version,
            "chunk_count": len(ids)
        }

    async def update_document_version(
            self,
            document_id: str,
            text: str,
            metadata: Optional[Dict[str, Any]] = None,
            version_notes: Optional[str] = None,
            requester_id: Optional[str] = None,
            status: str = "published"
    ) -> Dict[str, Any]:
        """
        Create a new version of an existing document (non-destructive update).
        """
        # Get latest version to determine parent
        latest = self.version_manager.get_latest_version(document_id)
        if not latest:
            raise ValueError(f"Document {document_id} not found")

        parent_version = latest["version"]
        source_name = latest["source_name"]

        # Calculate next version
        next_version = self._calculate_next_version(document_id)

        # Create new version
        result = await self.add_document_to_rag_local(
            source_name=source_name,
            text=text,
            metadata=metadata,
            document_id=document_id,
            version=next_version,
            parent_version=parent_version,
            status=status,
            version_notes=version_notes,
            created_by=requester_id
        )

        log_user_action(
            logger, "DOCUMENT_VERSION_UPDATE", requester_id,
            document_id=document_id, old_version=parent_version, new_version=next_version,
            status=status, has_notes=bool(version_notes)
        )
        return result

    async def get_document_version(
            self,
            document_id: str,
            version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific version of a document with its chunks.
        """
        # Get version record
        if version:
            version_info = self.version_manager.get_version(document_id, version)
        else:
            version_info = self.version_manager.get_latest_version(document_id)

        if not version_info:
            return None

        # Get chunks from ChromaDB
        try:

            chunk_ids = version_info["chunk_ids"]
            result = self.vector_store.get_documents_by_ids(chunk_ids)

            chunks = result.get("documents", [])
            metadatas = result.get("metadatas", [])

            return {
                "document_id": document_id,
                "version": version_info["version"],
                "source_name": version_info["source_name"],
                "chunks": chunks,
                "metadatas": metadatas,
                "created_at": version_info["created_at"],
                "created_by": version_info["created_by"],
                "status": version_info["status"],
                "version_notes": version_info["version_notes"],
                "parent_version": version_info["parent_version"]
            }
        except Exception as e:
            logger.exception("Failed to retrieve document version: %s", e)
            return None

    async def compare_document_versions(
            self,
            document_id: str,
            version1: str,
            version2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two versions of a document.
        """
        # Get both versions
        v1_data = await self.get_document_version(document_id, version1)
        v2_data = await self.get_document_version(document_id, version2)

        if not v1_data or not v2_data:
            return None

        # Combine chunks into full text
        text1 = "\n\n".join(v1_data["chunks"])
        text2 = "\n\n".join(v2_data["chunks"])

        # Compute diff
        import difflib
        diff = difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            fromfile=f"Version {version1}",
            tofile=f"Version {version2}",
            lineterm=''
        )
        diff_text = ''.join(diff)

        # Calculate statistics
        added_lines = diff_text.count('\n+')
        removed_lines = diff_text.count('\n-')
        chunk_diff = len(v2_data["chunks"]) - len(v1_data["chunks"])

        return {
            "document_id": document_id,
            "version1": version1,
            "version2": version2,
            "diff": diff_text,
            "summary": {
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "chunk_difference": chunk_diff,
                "v1_chunks": len(v1_data["chunks"]),
                "v2_chunks": len(v2_data["chunks"])
            },
            "version1_info": {
                "created_at": v1_data["created_at"],
                "created_by": v1_data["created_by"],
                "notes": v1_data["version_notes"]
            },
            "version2_info": {
                "created_at": v2_data["created_at"],
                "created_by": v2_data["created_by"],
                "notes": v2_data["version_notes"]
            }
        }

    async def list_documents(
            self,
            department: Optional[str] = None,
            status: Optional[str] = None,
            latest_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all documents with optional filtering.
        """
        # Get all documents from version tracking
        if status:
            documents = self.version_manager.get_documents_by_status(status)
        else:
            documents = self.version_manager.list_all_documents(latest_only=latest_only)

        # Filter by department if specified
        if department:
            filtered = []
            for doc in documents:
                if doc.get("metadata", {}).get("department") == department:
                    filtered.append(doc)
            documents = filtered

        return documents

    async def archive_document_version(
            self,
            document_id: str,
            version: str
    ) -> bool:
        """
        Archive (soft-delete) a specific version of a document.
        """
        try:
            # Update version tracking status
            success = self.version_manager.update_version_status(document_id, version, "archived")

            if not success:
                return False

            # Update ChromaDB metadata
            version_info = self.version_manager.get_version(document_id, version)
            if not version_info:
                return False

            chunk_ids = version_info["chunk_ids"]

            self.vector_store.update_metadatas(  ids=chunk_ids,
                                               metadata={"status": "archived", "is_latest_version": False})

            log_user_action(
                logger, "DOCUMENT_VERSION_ARCHIVED", "system",
                document_id=document_id, version=version
            )
            return True
        except Exception as e:
            logger.exception("Failed to archive document version: %s", e)
            return False

    async def seed_from_file(self, file_path: Optional[str] = None, source_name: Optional[str] = None,
                             force_reseed: bool = False) -> List[str]:
        """
        Read the given file or directory and index it.
        """

        # NEW DEFAULT PATH: data/companyData
        default_path = settings.TRAINING_DATA_DIR / "company"
        path = Path(file_path) if file_path else default_path
        logger.info("looking for path for data %s", path)
        if not path.exists():
            logger.warning("Seed path not found at %s", path)
            return []

        try:
            data = self.vector_store.get_collection_data()
            SHOW_DATA = True  # Just for debugging purpose
            if SHOW_DATA:
                pass
            has_data = not is_collection_empty(data)
        except Exception as e:
            logger.warning("Could not check collection size: %s. Assuming zero.", e)
            has_data = False
            data = {"ids": []}

        logger.info("has_data %s", has_data)
        if has_data and not force_reseed:
            logger.info(
                "Collection already contains %d documents. Skipping seed on startup (use /seed?reseed=true to force).",
                len(data.get("ids")))
            return []

        added_ids: List[str] = []

        # If path is a directory, check if it contains version subdirectories
        if path.is_dir():
            if force_reseed:
                logger.warning("Force re-seeding entire directory: %s. This may create duplicate chunks.", path)

            logger.info("Seeding directory: %s", path)

            # Check if this directory contains version subdirectories (v1, v2, v3, etc.)
            # Filter for directories starting with 'v' followed by a number
            version_dirs = []
            for d in path.iterdir():
                if d.is_dir() and d.name.lower().startswith('v'):
                    try:
                        # Verify the rest is a number (e.g., "1", "1.0", "2.5")
                        float(d.name[1:])
                        version_dirs.append(d)
                    except ValueError:
                        continue

            if version_dirs:
                # Sort version directories numerically (v2 < v10)
                def get_version_float(d):
                    try:
                        return float(d.name[1:])
                    except ValueError:
                        return 0.0

                sorted_version_dirs = sorted(version_dirs, key=get_version_float)
                logger.info("Found %d version directories in %s. Processing order: %s",
                            len(sorted_version_dirs), path, [d.name for d in sorted_version_dirs])

                category = path.name  # e.g., "company", "mission", etc.

                # Track the last seen version for each document to correctly link parents
                # Map: document_base_name -> version_string
                latest_versions_map = {}

                for version_dir in sorted_version_dirs:
                    version_str = version_dir.name[1:]  # Remove 'v' prefix
                    # Normalize version to semantic format (e.g., "1" -> "1.0")
                    if '.' not in version_str:
                        version_str = f"{version_str}.0"

                    logger.info("Processing version directory: %s (version %s)", version_dir.name, version_str)

                    for file_path in sorted(version_dir.iterdir()):
                        # Skip .meta.json files (they're companions, not documents)
                        if file_path.suffix == '.json' and file_path.stem.endswith('.meta'):
                            continue

                        if file_path.is_file():
                            try:
                                # Use doc_parser to read and parse file
                                text = parse_file(str(file_path))

                                # Generate document_id based on category + filename (same across versions)
                                doc_base_name = file_path.stem  # filename without extension
                                document_id = self._generate_document_id(f"{category}/{doc_base_name}")

                                # Source name for display
                                src_name = f"{category}/{version_dir.name}/{file_path.name}"

                                # Load custom metadata from companion .meta.json file
                                meta_file = file_path.with_suffix('.meta.json')
                                custom_meta = {}
                                if meta_file.exists():
                                    try:
                                        import json
                                        custom_meta = json.loads(meta_file.read_text(encoding='utf-8'))
                                        logger.info("Loaded metadata from %s", meta_file.name)
                                    except Exception as e:
                                        logger.warning("Failed to load metadata from %s: %s", meta_file.name, e)

                                # Merge metadata: custom metadata takes precedence
                                metadata = {
                                    "seeded": True,
                                    "category": category,
                                    **custom_meta  # Merge custom metadata
                                }

                                # Determine parent version dynamically
                                # If we've seen this doc before, that's the parent.
                                # If not, and this isn't v1.0, parent is None (it's a new doc introduced in a later version)
                                parent_version = latest_versions_map.get(doc_base_name)

                                result = await self.add_document_to_rag_local(
                                    source_name=src_name,
                                    text=text,
                                    chunks=None,
                                    metadata=metadata,  # Use merged metadata
                                    document_id=document_id,
                                    version=version_str,
                                    parent_version=parent_version,
                                    status="published",
                                    created_by="system_seed"
                                )

                                if result and result.get("ids"):
                                    added_ids.extend(result["ids"])
                                    log_user_action(
                                        logger, "SEED_FILE_PROCESSED", "system_seed",
                                        filename=file_path.name, version=version_str,
                                        chunk_count=result["chunk_count"], document_id=document_id,
                                        parent_version=parent_version, category=category
                                    )

                                # Update the map so the next version knows this is the parent
                                latest_versions_map[doc_base_name] = version_str

                            except Exception as e:
                                logger.exception("Failed to seed file %s: %s", file_path, e)
                                continue
            else:
                # Backward compatibility: process files directly in directory (old behavior)
                logger.info("No version directories found, processing files directly")
                for child in sorted(path.iterdir()):
                    if child.is_file():
                        try:
                            # Use doc_parser to read and parse file
                            text = parse_file(str(child))
                            # Use relative path + name for source_name for better uniqueness
                            src_name = str(child.relative_to(path.parent))
                            result = await self.add_document_to_rag_local(source_name=src_name, text=text, chunks=None,
                                                                          metadata={"seeded": True})
                            if result and result.get("ids"):
                                added_ids.extend(result["ids"])
                                log_user_action(
                                    logger, "SEED_FILE_LEGACY", "system_seed",
                                    filename=child.name, chunk_count=result["chunk_count"],
                                    version=result["version"])
                        except Exception as e:
                            logger.exception("Failed to seed file %s: %s", child, e)
                            continue
            return added_ids

        # Otherwise, it's a single file; ingest it. (Old behavior, primarily for backward compatibility)
        try:
            text = parse_file(str(path))
        except Exception as e:
            logger.exception("Failed to read seed file %s: %s", path, e)
            return []

        name = source_name or path.name
        try:
            result = await self.add_document_to_rag_local(source_name=name, text=text, chunks=None,
                                                          metadata={"seeded": True})
            if result and result.get("ids"):
                added_ids.extend(result["ids"])
                log_user_action(
                    logger, "SEED_SINGLE_FILE", "system_seed",
                    filename=path.name, chunk_count=result["chunk_count"],
                    version=result["version"])
        except Exception as e:
            logger.exception("Failed to seed file %s: %s", path, e)

        return added_ids

    def update_metadata(self, ids: List[str], metadata: Dict[str, Any]) -> bool:
        """
        Wrapper that updates metadata for existing ids using chroma_utils.update_metadatas.
        """

        sanitized = sanitize_metadata_dict(metadata)
        return self.vector_store.update_metadatas( ids=ids, metadata=sanitized)

    def clear_collection(self) -> None:
        """
        Delete all documents from the collection. Use with caution.
        """

        try:
            self.vector_store.delete_all_documents( )
        except Exception as e:
            logger.exception("Error clearing collection: %s", e)
            raise
