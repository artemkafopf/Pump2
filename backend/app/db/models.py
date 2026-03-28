from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_section: Mapped[str] = mapped_column(String(64), nullable=False, default="fact_epu")
    storage_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    selected_features_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    records: Mapped[list["Record"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="Record.row_index",
    )
    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="TrainedModel.created_at.desc()",
    )
    column_matches: Mapped[list["DatasetColumnMatch"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetColumnMatch.source_column",
    )
    generated_reports: Mapped[list["GeneratedReport"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="GeneratedReport.created_at.desc()",
    )


class Record(Base):
    __tablename__ = "records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="records")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    feature_columns_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    numeric_feature_columns_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dataset: Mapped[Dataset] = relationship(back_populates="trained_models")


class CanonicalVariable(Base):
    __tablename__ = "canonical_variables"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    aliases: Mapped[list["VariableAlias"]] = relationship(
        back_populates="canonical_variable",
        cascade="all, delete-orphan",
        order_by="VariableAlias.alias_name",
    )
    dataset_matches: Mapped[list["DatasetColumnMatch"]] = relationship(
        back_populates="canonical_variable",
        cascade="all, delete-orphan",
    )


class VariableAlias(Base):
    __tablename__ = "variable_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    canonical_variable_id: Mapped[int] = mapped_column(ForeignKey("canonical_variables.id"), nullable=False, index=True)
    alias_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    normalized_alias: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    canonical_variable: Mapped[CanonicalVariable] = relationship(back_populates="aliases")


class DatasetColumnMatch(Base):
    __tablename__ = "dataset_column_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    canonical_variable_id: Mapped[int | None] = mapped_column(
        ForeignKey("canonical_variables.id"),
        nullable=True,
        index=True,
    )
    source_column: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="suggested")
    llm_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dataset: Mapped[Dataset] = relationship(back_populates="column_matches")
    canonical_variable: Mapped[CanonicalVariable] = relationship(back_populates="dataset_matches")


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    trained_model_id: Mapped[int | None] = mapped_column(ForeignKey("trained_models.id"), nullable=True, index=True)
    report_type: Mapped[str] = mapped_column(String(64), nullable=False, default="analysis_forecast")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    llm_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dataset: Mapped[Dataset] = relationship(back_populates="generated_reports")
    trained_model: Mapped[TrainedModel | None] = relationship()
