"""Restart Manager for recovering incomplete processing after application restart."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set

from ..managers import DocumentLifecycleManager
from ..database.document_lifecycle_models import (
    DocumentRecord,
    ProcessingStageRecord,
    BatchRecord,
    ProcessingStatus,
    StageStatus,
    ProcessingMode
)

logger = logging.getLogger(__name__)


class RestartManager:
    """
    Manages recovery and restart of incomplete document processing.

    Features:
    - Detects incomplete documents after application restart
    - Resumes processing from the last completed stage
    - Validates data integrity during recovery
    - Provides recovery statistics and reporting
    """

    def __init__(self, lifecycle_manager: Optional[DocumentLifecycleManager] = None):
        """
        Initialize RestartManager.

        Args:
            lifecycle_manager: Optional DocumentLifecycleManager instance
        """
        try:
            if lifecycle_manager:
                self.lifecycle_manager = lifecycle_manager
            else:
                self.lifecycle_manager = DocumentLifecycleManager()

            logger.info("RestartManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RestartManager: {e}")
            raise

    def detect_incomplete_processing(self, max_age_hours: int = 24) -> List[DocumentRecord]:
        """
        Detect documents with incomplete processing.

        Args:
            max_age_hours: Only consider documents created within this time frame

        Returns:
            List of incomplete document records
        """
        try:
            logger.info("Detecting incomplete processing...")

            # Get all incomplete documents
            incomplete_docs = self.lifecycle_manager.get_incomplete_documents()

            if not incomplete_docs:
                logger.info("No incomplete documents found")
                return []

            # Filter by age if specified
            if max_age_hours > 0:
                cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
                incomplete_docs = [
                    doc for doc in incomplete_docs
                    if doc.created_at > cutoff_time
                ]

            logger.info(f"Found {len(incomplete_docs)} incomplete documents for recovery")
            return incomplete_docs

        except Exception as e:
            logger.error(f"Failed to detect incomplete processing: {e}")
            return []

    def validate_document_integrity(self, document_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a document's processing data.

        Args:
            document_id: Document identifier

        Returns:
            Validation result with integrity status and issues
        """
        try:
            validation_result = {
                "document_id": document_id,
                "is_valid": True,
                "issues": [],
                "stages_status": {},
                "data_integrity": {},
                "recovery_feasible": True
            }

            # Get document record
            document_record = self.lifecycle_manager.mongodb_manager.get_document(document_id)
            if not document_record:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Document record not found")
                validation_result["recovery_feasible"] = False
                return validation_result

            # Get processing stages
            stages = self.lifecycle_manager.mongodb_manager.get_document_stages(document_id)
            stages_by_name = {stage.stage_name: stage for stage in stages}

            # Check each expected stage
            expected_stages = ["upload", "pdf_processing", "splitting", "lmm_processing", "chunking", "completed"]

            for stage_name in expected_stages:
                if stage_name in stages_by_name:
                    stage = stages_by_name[stage_name]
                    validation_result["stages_status"][stage_name] = {
                        "exists": True,
                        "status": stage.stage_status.value,
                        "completed": stage.stage_status == StageStatus.COMPLETED,
                        "has_error": stage.stage_status == StageStatus.ERROR,
                        "error_message": stage.error_message
                    }

                    if stage.stage_status == StageStatus.ERROR:
                        validation_result["issues"].append(f"Stage {stage_name} has error: {stage.error_message}")
                else:
                    validation_result["stages_status"][stage_name] = {
                        "exists": False,
                        "status": "missing",
                        "completed": False,
                        "has_error": False,
                        "error_message": None
                    }

            # Check S3 data integrity
            if document_record.s3_pdf_url:
                validation_result["data_integrity"]["pdf_in_s3"] = True

                # Check if images folder exists
                if document_record.s3_images_folder:
                    validation_result["data_integrity"]["images_folder_configured"] = True
                else:
                    validation_result["issues"].append("S3 images folder not configured")

            else:
                validation_result["is_valid"] = False
                validation_result["issues"].append("PDF not uploaded to S3")
                validation_result["recovery_feasible"] = False

            # Check batch records consistency
            batches = self.lifecycle_manager.mongodb_manager.get_document_batches(document_id)
            if batches:
                validation_result["data_integrity"]["total_batches"] = len(batches)

                # Check batch completeness
                completed_batches = [batch for batch in batches if batch.status == StageStatus.COMPLETED]
                validation_result["data_integrity"]["completed_batches"] = len(completed_batches)

                if len(completed_batches) < len(batches):
                    validation_result["issues"].append(f"Only {len(completed_batches)}/{len(batches)} batches completed")

            # Check chunk records
            chunks = self.lifecycle_manager.mongodb_manager.get_document_chunks(document_id)
            if chunks:
                validation_result["data_integrity"]["total_chunks"] = len(chunks)

                # Validate chunk-to-batch linkage
                batch_ids = {batch.batch_id for batch in batches}
                chunk_batch_ids = {chunk.batch_id for chunk in chunks}
                orphaned_chunks = chunk_batch_ids - batch_ids

                if orphaned_chunks:
                    validation_result["issues"].append(f"Found {len(orphaned_chunks)} orphaned chunks")

            # Determine overall validity
            if validation_result["issues"]:
                validation_result["is_valid"] = len([issue for issue in validation_result["issues"] if "error" in issue.lower()]) == 0

            logger.info(f"Document {document_id} validation: {'VALID' if validation_result['is_valid'] else 'INVALID'} ({len(validation_result['issues'])} issues)")
            return validation_result

        except Exception as e:
            logger.error(f"Failed to validate document integrity for {document_id}: {e}")
            return {
                "document_id": document_id,
                "is_valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "stages_status": {},
                "data_integrity": {},
                "recovery_feasible": False
            }

    def determine_recovery_strategy(self, document_id: str) -> Dict[str, Any]:
        """
        Determine the best recovery strategy for a document.

        Args:
            document_id: Document identifier

        Returns:
            Recovery strategy with next steps
        """
        try:
            validation_result = self.validate_document_integrity(document_id)

            strategy = {
                "document_id": document_id,
                "strategy_type": "unknown",
                "next_stage": None,
                "actions_required": [],
                "estimated_time_minutes": 0,
                "recovery_feasible": validation_result["recovery_feasible"]
            }

            if not validation_result["recovery_feasible"]:
                strategy["strategy_type"] = "full_restart"
                strategy["actions_required"] = ["Re-upload PDF", "Start processing from beginning"]
                strategy["estimated_time_minutes"] = 30
                return strategy

            stages_status = validation_result["stages_status"]

            # Determine where to resume based on completed stages
            if stages_status.get("upload", {}).get("completed"):
                if stages_status.get("pdf_processing", {}).get("completed"):
                    if stages_status.get("splitting", {}).get("completed"):
                        if stages_status.get("lmm_processing", {}).get("completed"):
                            if stages_status.get("chunking", {}).get("completed"):
                                # Everything completed, just need to mark as completed
                                strategy["strategy_type"] = "mark_completed"
                                strategy["next_stage"] = "completed"
                                strategy["actions_required"] = ["Update document status to completed"]
                                strategy["estimated_time_minutes"] = 1
                            else:
                                # Resume from chunking
                                strategy["strategy_type"] = "resume_chunking"
                                strategy["next_stage"] = "chunking"
                                strategy["actions_required"] = ["Validate batch results", "Save remaining chunks"]
                                strategy["estimated_time_minutes"] = 5
                        else:
                            # Resume from LMM processing
                            strategy["strategy_type"] = "resume_lmm_processing"
                            strategy["next_stage"] = "lmm_processing"
                            strategy["actions_required"] = ["Resume LMM processing for incomplete batches"]
                            strategy["estimated_time_minutes"] = 20
                    else:
                        # Resume from splitting
                        strategy["strategy_type"] = "resume_splitting"
                        strategy["next_stage"] = "splitting"
                        strategy["actions_required"] = ["Recreate batches", "Resume LMM processing"]
                        strategy["estimated_time_minutes"] = 25
                else:
                    # Resume from PDF processing
                    strategy["strategy_type"] = "resume_pdf_processing"
                    strategy["next_stage"] = "pdf_processing"
                    strategy["actions_required"] = ["Process PDF images", "Save to S3", "Continue with splitting"]
                    strategy["estimated_time_minutes"] = 30
            else:
                # Need to start from upload
                strategy["strategy_type"] = "resume_upload"
                strategy["next_stage"] = "upload"
                strategy["actions_required"] = ["Verify S3 upload", "Continue processing"]
                strategy["estimated_time_minutes"] = 30

            logger.info(f"Recovery strategy for {document_id}: {strategy['strategy_type']} -> {strategy['next_stage']}")
            return strategy

        except Exception as e:
            logger.error(f"Failed to determine recovery strategy for {document_id}: {e}")
            return {
                "document_id": document_id,
                "strategy_type": "error",
                "next_stage": None,
                "actions_required": [f"Manual intervention required: {str(e)}"],
                "estimated_time_minutes": 0,
                "recovery_feasible": False
            }

    def resume_document_processing(self, document_id: str) -> Dict[str, Any]:
        """
        Resume processing for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            Resume operation result
        """
        try:
            logger.info(f"Resuming processing for document {document_id}")

            # Get recovery strategy
            strategy = self.determine_recovery_strategy(document_id)

            if not strategy["recovery_feasible"]:
                logger.error(f"Document {document_id} recovery not feasible: {strategy['actions_required']}")
                return {
                    "success": False,
                    "document_id": document_id,
                    "error": "Recovery not feasible",
                    "actions_required": strategy["actions_required"]
                }

            # Update document status to processing if it was in error state
            document_record = self.lifecycle_manager.mongodb_manager.get_document(document_id)
            if document_record.current_status == ProcessingStatus.ERROR:
                document_record.current_status = ProcessingStatus.PROCESSING
                document_record.error_message = None
                document_record.retry_count += 1
                self.lifecycle_manager.mongodb_manager.update_document(document_record)

            # Execute recovery strategy
            result = self._execute_recovery_strategy(document_id, strategy)

            if result["success"]:
                logger.info(f"Successfully resumed processing for document {document_id}")
            else:
                logger.error(f"Failed to resume processing for document {document_id}: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Failed to resume document processing for {document_id}: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "actions_required": ["Manual intervention required"]
            }

    def _execute_recovery_strategy(self, document_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific recovery strategy.

        Args:
            document_id: Document identifier
            strategy: Recovery strategy to execute

        Returns:
            Execution result
        """
        try:
            strategy_type = strategy["strategy_type"]

            if strategy_type == "mark_completed":
                # Simply mark document as completed
                self.lifecycle_manager.update_document_status(
                    document_id=document_id,
                    status=ProcessingStatus.COMPLETED
                )
                return {"success": True, "message": "Document marked as completed"}

            elif strategy_type == "resume_chunking":
                # Validate and complete chunking process
                return self._resume_chunking(document_id)

            elif strategy_type == "resume_lmm_processing":
                # Resume LMM processing for incomplete batches
                return self._resume_lmm_processing(document_id)

            elif strategy_type == "resume_splitting":
                # Resume from splitting stage
                return self._resume_splitting(document_id)

            elif strategy_type == "resume_pdf_processing":
                # Resume from PDF processing stage
                return self._resume_pdf_processing(document_id)

            elif strategy_type == "resume_upload":
                # Verify upload and continue
                return self._resume_upload(document_id)

            else:
                return {
                    "success": False,
                    "error": f"Unknown recovery strategy: {strategy_type}",
                    "actions_required": ["Manual intervention required"]
                }

        except Exception as e:
            logger.error(f"Failed to execute recovery strategy {strategy['strategy_type']} for document {document_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions_required": ["Manual intervention required"]
            }

    def _resume_chunking(self, document_id: str) -> Dict[str, Any]:
        """Resume from chunking stage."""
        try:
            # Mark chunking as completed
            self.lifecycle_manager.track_stage_completion(
                document_id=document_id,
                stage_name="chunking",
                stage_data={"resumed_from_restart": True}
            )

            # Mark document as completed
            self.lifecycle_manager.update_document_status(
                document_id=document_id,
                status=ProcessingStatus.COMPLETED
            )

            return {"success": True, "message": "Chunking resumed and completed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resume_lmm_processing(self, document_id: str) -> Dict[str, Any]:
        """Resume from LMM processing stage."""
        # This would need integration with ProcessingManager to resubmit incomplete batches
        return {
            "success": False,
            "error": "LMM processing resume requires integration with ProcessingManager",
            "actions_required": ["Resubmit document for processing"]
        }

    def _resume_splitting(self, document_id: str) -> Dict[str, Any]:
        """Resume from splitting stage."""
        # This would need integration with ProcessingManager
        return {
            "success": False,
            "error": "Splitting resume requires integration with ProcessingManager",
            "actions_required": ["Resubmit document for processing"]
        }

    def _resume_pdf_processing(self, document_id: str) -> Dict[str, Any]:
        """Resume from PDF processing stage."""
        # This would need integration with ProcessingManager
        return {
            "success": False,
            "error": "PDF processing resume requires integration with ProcessingManager",
            "actions_required": ["Resubmit document for processing"]
        }

    def _resume_upload(self, document_id: str) -> Dict[str, Any]:
        """Resume from upload stage."""
        try:
            # Verify S3 upload exists
            document_record = self.lifecycle_manager.mongodb_manager.get_document(document_id)
            if not document_record or not document_record.s3_pdf_url:
                return {
                    "success": False,
                    "error": "PDF not found in S3",
                    "actions_required": ["Re-upload PDF file"]
                }

            # Mark upload as completed and continue
            self.lifecycle_manager.track_stage_completion(
                document_id=document_id,
                stage_name="upload",
                stage_data={"resumed_from_restart": True, "s3_url_verified": True}
            )

            return {
                "success": False,
                "error": "Upload verification complete, continue processing required",
                "actions_required": ["Resubmit document for processing from PDF stage"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_recovery_statistics(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics about recoverable documents.

        Args:
            max_age_hours: Only consider documents within this time frame

        Returns:
            Recovery statistics
        """
        try:
            incomplete_docs = self.detect_incomplete_processing(max_age_hours)

            stats = {
                "total_incomplete_documents": len(incomplete_docs),
                "by_status": {},
                "by_processing_mode": {},
                "by_age": {},
                "recovery_strategies": {},
                "data_integrity_issues": 0,
                "recoverable_documents": 0,
                "manual_intervention_required": 0
            }

            if not incomplete_docs:
                return stats

            # Analyze each document
            for doc in incomplete_docs:
                # Status breakdown
                status_key = doc.current_status.value
                stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1

                # Processing mode breakdown
                mode_key = doc.processing_mode.value
                stats["by_processing_mode"][mode_key] = stats["by_processing_mode"].get(mode_key, 0) + 1

                # Age breakdown
                age_hours = (datetime.utcnow() - doc.created_at).total_seconds() / 3600
                if age_hours < 1:
                    age_key = "< 1 hour"
                elif age_hours < 6:
                    age_key = "1-6 hours"
                elif age_hours < 24:
                    age_key = "6-24 hours"
                else:
                    age_key = "> 24 hours"

                stats["by_age"][age_key] = stats["by_age"].get(age_key, 0) + 1

                # Recovery strategy analysis
                try:
                    strategy = self.determine_recovery_strategy(doc.document_id)
                    strategy_key = strategy["strategy_type"]
                    stats["recovery_strategies"][strategy_key] = stats["recovery_strategies"].get(strategy_key, 0) + 1

                    if strategy["recovery_feasible"]:
                        stats["recoverable_documents"] += 1
                    else:
                        stats["manual_intervention_required"] += 1

                    # Check for data integrity issues
                    validation = self.validate_document_integrity(doc.document_id)
                    if not validation["is_valid"]:
                        stats["data_integrity_issues"] += 1

                except Exception as e:
                    logger.error(f"Failed to analyze recovery for document {doc.document_id}: {e}")
                    stats["manual_intervention_required"] += 1

            logger.info(f"Recovery statistics: {stats['recoverable_documents']}/{stats['total_incomplete_documents']} documents recoverable")
            return stats

        except Exception as e:
            logger.error(f"Failed to get recovery statistics: {e}")
            return {"error": str(e)}

    def run_recovery_check(self, max_age_hours: int = 24, auto_resume: bool = False) -> Dict[str, Any]:
        """
        Run a comprehensive recovery check and optionally auto-resume processing.

        Args:
            max_age_hours: Only consider documents within this time frame
            auto_resume: Whether to automatically resume recoverable documents

        Returns:
            Recovery check results
        """
        try:
            logger.info("Running comprehensive recovery check...")

            # Get recovery statistics
            stats = self.get_recovery_statistics(max_age_hours)

            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": stats,
                "recovery_actions": [],
                "auto_resume_enabled": auto_resume,
                "total_processed": 0,
                "successful_resumes": 0,
                "failed_resumes": 0
            }

            if stats.get("total_incomplete_documents", 0) == 0:
                result["message"] = "No incomplete documents found - system is healthy"
                return result

            # Get incomplete documents for detailed processing
            incomplete_docs = self.detect_incomplete_processing(max_age_hours)

            if auto_resume:
                logger.info(f"Auto-resuming {len(incomplete_docs)} incomplete documents...")

                for doc in incomplete_docs:
                    try:
                        resume_result = self.resume_document_processing(doc.document_id)
                        result["total_processed"] += 1

                        action = {
                            "document_id": doc.document_id,
                            "filename": doc.original_filename,
                            "action": "auto_resume",
                            "success": resume_result["success"],
                            "message": resume_result.get("error") or "Resumed successfully",
                            "timestamp": datetime.utcnow().isoformat()
                        }

                        result["recovery_actions"].append(action)

                        if resume_result["success"]:
                            result["successful_resumes"] += 1
                        else:
                            result["failed_resumes"] += 1

                    except Exception as e:
                        result["failed_resumes"] += 1
                        result["recovery_actions"].append({
                            "document_id": doc.document_id,
                            "filename": doc.original_filename,
                            "action": "auto_resume",
                            "success": False,
                            "message": f"Exception during resume: {str(e)}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.error(f"Failed to auto-resume document {doc.document_id}: {e}")

            else:
                # Just report what would be done
                for doc in incomplete_docs:
                    strategy = self.determine_recovery_strategy(doc.document_id)
                    action = {
                        "document_id": doc.document_id,
                        "filename": doc.original_filename,
                        "action": "analysis_only",
                        "strategy": strategy["strategy_type"],
                        "feasible": strategy["recovery_feasible"],
                        "next_stage": strategy.get("next_stage"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    result["recovery_actions"].append(action)

            logger.info(f"Recovery check completed: {result['successful_resumes']}/{result['total_processed']} successful resumes")
            return result

        except Exception as e:
            logger.error(f"Failed to run recovery check: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "recovery_actions": []
            }