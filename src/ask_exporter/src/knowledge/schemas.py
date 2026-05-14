"""Pydantic data models for BOM items, export control results, and the unified report."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class HardwareItem(BaseModel):
    category: str = "hardware"
    name: str
    subcategory: str = "other"
    specifications: dict[str, Any] = Field(default_factory=dict)
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    quantity: Optional[str] = None
    estimated_cost: Optional[str] = None
    notes: str = ""


class SoftwareItem(BaseModel):
    category: str = "software"
    name: str
    version: Optional[str] = None
    purpose: str = ""
    license: Optional[str] = None
    url: Optional[str] = None


class MaterialItem(BaseModel):
    category: str = "materials"
    name: str
    subcategory: str = "other"
    specification: Optional[str] = None
    quantity: Optional[str] = None
    supplier: Optional[str] = None
    estimated_cost: Optional[str] = None


class BOM(BaseModel):
    hardware: list[HardwareItem] = Field(default_factory=list)
    software: list[SoftwareItem] = Field(default_factory=list)
    materials: list[MaterialItem] = Field(default_factory=list)

    def hardware_names(self) -> list[str]:
        return [item.name for item in self.hardware]

    def items_for_export_check(self, include_software: bool = False, include_materials: bool = False) -> list[str]:
        """Return item names for export control checking. Hardware only by default."""
        items = self.hardware_names()
        if include_software:
            items += [item.name for item in self.software]
        if include_materials:
            items += [item.name for item in self.materials]
        return items

    @classmethod
    def from_dict(cls, data: dict) -> "BOM":
        return cls(
            hardware=[HardwareItem(**item) for item in data.get("hardware", [])],
            software=[SoftwareItem(**item) for item in data.get("software", [])],
            materials=[MaterialItem(**item) for item in data.get("materials", [])],
        )


class ExportControlResult(BaseModel):
    item_name: str
    us_status: str = "unclear"       # controlled / not_controlled / unclear
    us_details: str = ""
    germany_status: str = "unclear"
    germany_details: str = ""
    eu_status: str = "unclear"
    eu_details: str = ""
    overall_risk: str = "unclear"    # high / medium / low / clear / unclear
    recommendation: str = ""

    def risk_color(self) -> str:
        return {"high": "red", "medium": "orange", "low": "yellow", "clear": "green"}.get(
            self.overall_risk, "grey"
        )


class UnifiedReport(BaseModel):
    input_type: str                                          # direct_items / arxiv / pdf
    paper_info: Optional[dict] = None
    bom: Optional[dict] = None                              # serialised BOM dict
    export_control_results: list[ExportControlResult] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: float = 0.0

    def to_json_dict(self) -> dict:
        return self.model_dump()
