"""Configuration Model schema — the contract between AI interpretation
and downstream enumeration/optimization.

Every AI-extracted field carries provenance (which ABOM row it came from)
and a confidence score, so an SME can review and approve it.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


SmeStatus = Literal["pending", "approved", "edited", "rejected"]


class Provenance(BaseModel):
    abom_rows: list[int] = Field(default_factory=list)
    source_text: Optional[str] = None
    rule: Optional[str] = None  # e.g. "find#_collision", "notes_phrase"


class RequiredGroup(BaseModel):
    group_id: str
    parts: list[str]
    qty_per_vehicle: int
    find_num: int
    provenance: Provenance = Provenance()
    confidence: float = 1.0
    sme_status: SmeStatus = "pending"


class OptionGroup(BaseModel):
    group_id: str
    find_num: int
    qty_per_vehicle: int
    select: Literal["exactly_one", "at_most_one"] = "exactly_one"
    choices: list[str]
    provenance: Provenance = Provenance()
    confidence: float = 1.0
    sme_status: SmeStatus = "pending"


class ChoiceRef(BaseModel):
    group: str
    choice: str


class Constraint(BaseModel):
    type: Literal["implies", "excludes", "requires_one_of", "forbidden_with"]
    if_: ChoiceRef = Field(alias="if")
    then: Optional[ChoiceRef] = None  # for implies
    excluded: Optional[ChoiceRef] = None  # for excludes / forbidden_with
    one_of: Optional[list[ChoiceRef]] = None  # for requires_one_of
    provenance: Provenance = Provenance()
    confidence: float = 0.0
    sme_status: SmeStatus = "pending"

    class Config:
        populate_by_name = True


class ConfigurationModel(BaseModel):
    vehicle_id: str
    abom_version: str = "v1"
    required_groups: list[RequiredGroup] = Field(default_factory=list)
    option_groups: list[OptionGroup] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    interpreter_meta: dict = Field(default_factory=dict)


class ValidConfiguration(BaseModel):
    """One concrete buildable configuration."""
    config_id: str
    vehicle_id: str
    choices: dict[str, str]   # group_id -> chosen part
    parts_list: dict[str, int]  # part_number -> qty per vehicle


class BuildPlanLine(BaseModel):
    config_id: str
    quantity: int
    choices: dict[str, str]


class BindingConstraint(BaseModel):
    part_number: str
    on_hand: int
    consumed: int
    slack: int


class UnlockSuggestion(BaseModel):
    part_number: str
    additional_qty_needed: int
    additional_vehicles_unlocked: int
    estimated_cost: Optional[float] = None
    vehicles_per_dollar: Optional[float] = None


class BuildPlanResult(BaseModel):
    total_vehicles: int
    plan: list[BuildPlanLine]
    parts_consumed: dict[str, int]
    binding_constraints: list[BindingConstraint]
    unlock_suggestions: list[UnlockSuggestion]
