"""Audit reporting for failure updates"""

import pandas as pd
import openpyxl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from .matching import DuplicateStatus


class AuditReport:
    """Audit report for failure update workflow"""
    
    def __init__(self):
        self.summary = {}
        self.new_records = []
        self.duplicates_skipped = []
        self.likely_duplicates = []
        self.field_fill_matrix = {}
        self.source_conflicts = []
        self.missing_fields = {}
        self.opz_matches = []
        self.techregime_matches = []
        self.telemetry_matches = []
        self.lab_chemistry_missing = []
        # Flat per-header column-resolution records from every ingested source.
        self.column_mappings: List[Dict] = []
        # Incoming rows within tolerance of an existing run (not appended).
        self.needs_review: List[Dict] = []
        # In-place field updates applied to existing target rows (old -> new).
        self.row_updates: List[Dict] = []
        # Physically-impossible values corrected on ingest (old -> new + reason).
        self.data_fixes: List[Dict] = []
    
    def add_summary_stat(self, key: str, value: any):
        """Add summary statistic"""
        self.summary[key] = value
    
    def add_new_record(self, record_data: Dict):
        """Add new record to audit"""
        self.new_records.append(record_data)
    
    def add_duplicate(self, record_data: Dict, status: DuplicateStatus):
        """Add duplicate record"""
        if status == DuplicateStatus.EXACT_EXISTING:
            self.duplicates_skipped.append(record_data)
        elif status == DuplicateStatus.LIKELY_EXISTING:
            self.likely_duplicates.append(record_data)
    
    def add_field_fill(self, field_name: str, count_filled: int, count_empty: int, source: Optional[str] = None):
        """Track field filling statistics"""
        self.field_fill_matrix[field_name] = {
            "filled": count_filled,
            "empty": count_empty,
            "source": source
        }
    
    def add_source_conflict(self, well: str, field: str, value1: any, value2: any, sources: List[str]):
        """Add field conflict between sources"""
        self.source_conflicts.append({
            "well": well,
            "field": field,
            "value1": value1,
            "value2": value2,
            "sources": sources
        })
    
    def add_missing_field(self, field_name: str, reason: str, count: int):
        """Add missing field"""
        if field_name not in self.missing_fields:
            self.missing_fields[field_name] = []
        self.missing_fields[field_name].append({
            "reason": reason,
            "count": count
        })
    
    def add_needs_review(self, new_row: Dict, target_row: Dict, reason: Optional[str] = None):
        """Record an incoming row that matched an existing run within tolerance.

        Both the incoming and the matched target row are shown (key identity
        fields) so a human can decide whether it is the same run with a corrected
        date or a genuinely new event.
        """
        def _pick(source: Dict, *keys):
            for key in keys:
                if key in source and source[key] is not None:
                    return source[key]
            return None

        self.needs_review.append({
            "reason": reason,
            "well": _pick(new_row, "well", "Скв.", "normalized_well"),
            "new_failure_date": _pick(new_row, "failure_date", "Дата остановки"),
            "target_failure_date": _pick(target_row, "Дата остановки", "failure_date"),
            "new_install_date": _pick(new_row, "installation_date", "Дата монтажа"),
            "target_install_date": _pick(target_row, "Дата монтажа", "installation_date"),
            "new_failure_reason": _pick(new_row, "failure_reason", "Причина остановки"),
            "target_failure_reason": _pick(target_row, "Причина остановки", "failure_reason"),
        })

    def add_data_fix(self, well, field, old_value, new_value, reason: str):
        """Record a physically-impossible value corrected on ingest."""
        self.data_fixes.append({
            "well": well,
            "field": field,
            "old": old_value,
            "new": new_value,
            "reason": reason,
        })

    def add_row_update(self, well, changes: List[Dict]):
        """Record in-place field updates (each ``{field, old, new}``) for one row."""
        for change in changes:
            self.row_updates.append({
                "well": well,
                "field": change.get("field"),
                "old": change.get("old"),
                "new": change.get("new"),
            })

    def add_column_mapping(self, report):
        """Record a source file's column-resolution report.

        ``report`` is a :class:`analysis.ingest.column_resolution.ColumnMappingReport`;
        its flat per-header records are appended so the audit workbook/markdown
        can show exactly how each source header was matched (or left UNMAPPED).
        """
        self.column_mappings.extend(report.to_records())

    def add_opz_match(self, well: str, opz_date: any, opz_type: str, days_before_failure: int):
        """Add OPZ match"""
        self.opz_matches.append({
            "well": well,
            "opz_date": opz_date,
            "opz_type": opz_type,
            "days_before_failure": days_before_failure
        })
    
    def to_excel(self, output_path: str):
        """
        Write audit report to Excel file.
        
        Creates multiple sheets:
        - summary
        - new_records
        - duplicates_skipped
        - likely_duplicates_review
        - field_fill_matrix
        - source_conflicts
        - missing_fields
        - opz_matches
        """
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_df = pd.DataFrame(
                list(self.summary.items()),
                columns=["Metric", "Value"]
            )
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            
            # New records sheet
            if self.new_records:
                new_df = pd.DataFrame(self.new_records)
                new_df.to_excel(writer, sheet_name="new_records", index=False)
            
            # Duplicates skipped
            if self.duplicates_skipped:
                dup_df = pd.DataFrame(self.duplicates_skipped)
                dup_df.to_excel(writer, sheet_name="duplicates_skipped", index=False)
            
            # Likely duplicates
            if self.likely_duplicates:
                likely_df = pd.DataFrame(self.likely_duplicates)
                likely_df.to_excel(writer, sheet_name="likely_duplicates_review", index=False)
            
            # Field fill matrix
            if self.field_fill_matrix:
                fill_df = pd.DataFrame(
                    self.field_fill_matrix
                ).T.reset_index()
                fill_df.columns = ["Field", "Filled", "Empty", "Source"]
                fill_df.to_excel(writer, sheet_name="field_fill_matrix", index=False)
            
            # Source conflicts
            if self.source_conflicts:
                conflict_df = pd.DataFrame(self.source_conflicts)
                conflict_df.to_excel(writer, sheet_name="source_conflicts", index=False)
            
            # Missing fields
            if self.missing_fields:
                missing_list = []
                for field, reasons in self.missing_fields.items():
                    for reason_info in reasons:
                        missing_list.append({
                            "Field": field,
                            "Reason": reason_info["reason"],
                            "Count": reason_info["count"]
                        })
                missing_df = pd.DataFrame(missing_list)
                missing_df.to_excel(writer, sheet_name="missing_fields", index=False)
            
            # OPZ matches
            if self.opz_matches:
                opz_df = pd.DataFrame(self.opz_matches)
                opz_df.to_excel(writer, sheet_name="opz_matches", index=False)

            # Column mapping report (how each source header was resolved)
            if self.column_mappings:
                mapping_df = pd.DataFrame(self.column_mappings)
                mapping_df.to_excel(writer, sheet_name="column_mappings", index=False)

            # Rows within tolerance of an existing run — for human review
            if self.needs_review:
                review_df = pd.DataFrame(self.needs_review)
                review_df.to_excel(writer, sheet_name="needs_review", index=False)

            # In-place updates applied to existing rows
            if self.row_updates:
                updates_df = pd.DataFrame(self.row_updates)
                updates_df.to_excel(writer, sheet_name="row_updates", index=False)

            # Physically-impossible values corrected on ingest
            if self.data_fixes:
                fixes_df = pd.DataFrame(self.data_fixes)
                fixes_df.to_excel(writer, sheet_name="data_fixes", index=False)
    
    def to_markdown(self, output_path: str):
        """
        Write audit report to Markdown file.
        
        Args:
            output_path: Output file path
        """
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# Failure Update Audit Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
        ]
        
        # Summary statistics
        for key, value in self.summary.items():
            lines.append(f"- **{key}**: {value}")
        
        lines.extend([
            "",
            "## Statistics",
            "",
            f"- **New records appended**: {len(self.new_records)}",
            f"- **Existing rows updated in place**: {len(set((u['well'], u['field']) for u in self.row_updates))} field change(s)",
            f"- **Rows needing review (within tolerance)**: {len(self.needs_review)}",
            f"- **Exact duplicates skipped**: {len(self.duplicates_skipped)}",
            f"- **Likely duplicates (review)**: {len(self.likely_duplicates)}",
            f"- **Source conflicts**: {len(self.source_conflicts)}",
            "",
            "## Fields Filled",
            "",
        ])
        
        # Field matrix
        for field, stats in self.field_fill_matrix.items():
            filled = stats.get("filled", 0)
            empty = stats.get("empty", 0)
            total = filled + empty
            pct = (filled / total * 100) if total > 0 else 0
            lines.append(f"- **{field}**: {filled}/{total} ({pct:.1f}%)")
            if stats.get("source"):
                lines.append(f"  - Source: {stats['source']}")
        
        lines.extend([
            "",
            "## Missing Fields",
            "",
        ])
        
        # Missing fields
        for field, reasons in self.missing_fields.items():
            lines.append(f"- **{field}**:")
            for reason_info in reasons:
                lines.append(f"  - {reason_info['reason']}: {reason_info['count']} records")
        
        lines.extend([
            "",
            "## Water Chemistry",
            "",
            "All water chemistry / lab columns left empty as sources are unavailable:",
            "",
        ])
        
        for col in self.lab_chemistry_missing:
            lines.append(f"- {col}")

        lines.append("")

        # Column mapping report -- surface only the non-exact resolutions and any
        # UNMAPPED headers, which are the ones worth a human's attention.
        noteworthy = [
            record for record in self.column_mappings
            if record.get("method") != "exact"
        ]
        if noteworthy:
            lines.extend([
                "## Column Mapping (non-exact resolutions)",
                "",
                "| Source | Header | Canonical | Method | Score |",
                "| --- | --- | --- | --- | --- |",
            ])
            for record in noteworthy:
                lines.append(
                    f"| {record.get('source_file') or record.get('dataset') or ''} "
                    f"| {record.get('source_header')} "
                    f"| {record.get('canonical')} "
                    f"| {record.get('method')} "
                    f"| {record.get('score')} |"
                )
            lines.append("")

        # Rows updated in place (old -> new) on an existing key.
        if self.row_updates:
            lines.extend([
                "## Rows Updated In Place",
                "",
                "| Well | Field | Old | New |",
                "| --- | --- | --- | --- |",
            ])
            for update in self.row_updates:
                lines.append(
                    f"| {update.get('well')} | {update.get('field')} "
                    f"| {update.get('old')} | {update.get('new')} |"
                )
            lines.append("")

        # Physically-impossible values corrected on ingest (old -> new).
        if self.data_fixes:
            lines.extend([
                "## Data Fixes (impossible values corrected)",
                "",
                "| Well | Field | Old | New | Reason |",
                "| --- | --- | --- | --- | --- |",
            ])
            for fix in self.data_fixes:
                lines.append(
                    f"| {fix.get('well')} | {fix.get('field')} | {fix.get('old')} "
                    f"| {fix.get('new')} | {fix.get('reason')} |"
                )
            lines.append("")

        # Incoming rows within tolerance of an existing run (not appended).
        if self.needs_review:
            lines.extend([
                "## Needs Review (within tolerance of an existing run)",
                "",
                "| Well | Reason | New failure date | Target failure date |",
                "| --- | --- | --- | --- |",
            ])
            for review in self.needs_review:
                lines.append(
                    f"| {review.get('well')} | {review.get('reason')} "
                    f"| {review.get('new_failure_date')} | {review.get('target_failure_date')} |"
                )
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def generate_field_fill_matrix(records: pd.DataFrame, target_columns: List[str]) -> Dict:
    """
    Generate field fill statistics for target columns.
    
    Args:
        records: Records dataframe
        target_columns: List of target column names
        
    Returns:
        Dictionary of field fill statistics
    """
    matrix = {}
    
    for col in target_columns:
        if col in records.columns:
            filled = records[col].notna().sum()
            empty = records[col].isna().sum()
            matrix[col] = {
                "filled": filled,
                "empty": empty,
                "source": None
            }
    
    return matrix


def compare_dataframes_for_conflicts(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_columns: List[str]
) -> List[Dict]:
    """
    Compare two dataframes to find conflicting values.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        key_columns: Columns to use as key for matching
        
    Returns:
        List of conflicts
    """
    conflicts = []
    
    # This is a placeholder - implement full comparison as needed
    
    return conflicts
