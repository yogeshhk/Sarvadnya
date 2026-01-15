"""
Floor Plan JSON Schema Definition
Based on IFC standards and Schema.org FloorPlan with extensions
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class RoomType(str, Enum):
    """Standard room types following DIN 277-2 classification"""
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    CORRIDOR = "corridor"
    BALCONY = "balcony"
    STORAGE = "storage"
    OFFICE = "office"
    LAUNDRY = "laundry"
    GARAGE = "garage"
    OTHER = "other"


class AdjacencyType(str, Enum):
    """Types of spatial relationships between rooms"""
    ACCESS = "access"  # Direct door connection
    ADJACENT = "adjacent"  # Shares wall
    VISUAL = "visual"  # Window/opening
    VERTICAL = "vertical"  # Different floors


class Dimensions(BaseModel):
    """Physical dimensions of a space"""
    length: float = Field(..., gt=0, description="Length in meters")
    width: float = Field(..., gt=0, description="Width in meters")
    height: Optional[float] = Field(None, gt=0, description="Height in meters")
    unit: str = Field("m", description="Unit of measurement")

    @property
    def area(self) -> float:
        """Calculate area from length and width"""
        return self.length * self.width


class RoomFeatures(BaseModel):
    """Features and fixtures within a room"""
    windows: int = Field(0, ge=0, description="Number of windows")
    doors: int = Field(1, ge=0, description="Number of doors")
    balcony: bool = Field(False, description="Has balcony access")
    ensuite: bool = Field(False, description="Has ensuite bathroom")
    walk_in_closet: bool = Field(False, description="Has walk-in closet")
    natural_light: bool = Field(True, description="Has natural light")


class Geometry(BaseModel):
    """Geometric representation of room boundaries"""
    vertices: List[List[float]] = Field(
        ..., 
        description="List of [x, y] coordinates defining room boundary"
    )
    coordinates: str = Field("cartesian", description="Coordinate system")
    
    @validator('vertices')
    def validate_vertices(cls, v):
        """Ensure at least 3 vertices for a valid polygon"""
        if len(v) < 3:
            raise ValueError("Room must have at least 3 vertices")
        for vertex in v:
            if len(vertex) != 2:
                raise ValueError("Each vertex must have exactly 2 coordinates [x, y]")
        return v


class Room(BaseModel):
    """Individual room specification"""
    id: str = Field(..., description="Unique room identifier")
    type: RoomType = Field(..., description="Room type/function")
    name: Optional[str] = Field(None, description="Custom room name")
    area: float = Field(..., gt=0, description="Room area in square meters")
    dimensions: Dimensions = Field(..., description="Room dimensions")
    features: RoomFeatures = Field(default_factory=RoomFeatures)
    geometry: Optional[Geometry] = Field(None, description="Room boundary geometry")
    din_code: Optional[str] = Field(
        None, 
        description="DIN 277-2 classification code"
    )
    floor_level: int = Field(0, description="Floor level (0=ground)")
    
    @validator('area')
    def validate_area(cls, v, values):
        """Check if area matches dimensions"""
        if 'dimensions' in values:
            calculated_area = values['dimensions'].area
            if abs(v - calculated_area) > 0.1:  # Allow 0.1m² tolerance
                raise ValueError(
                    f"Area {v} doesn't match dimensions {calculated_area}"
                )
        return v


class Adjacency(BaseModel):
    """Relationship between two rooms"""
    room1: str = Field(..., description="First room ID")
    room2: str = Field(..., description="Second room ID")
    type: AdjacencyType = Field(..., description="Type of adjacency")
    has_wall: bool = Field(True, description="Shares a wall")
    has_door: bool = Field(False, description="Has door connection")
    description: Optional[str] = Field(None, description="Additional details")


class Zone(BaseModel):
    """Functional grouping of rooms"""
    name: str = Field(..., description="Zone name (e.g., 'sleeping_zone')")
    rooms: List[str] = Field(..., description="List of room IDs in this zone")
    separation_level: str = Field(
        "public", 
        description="Privacy level: public, semi-private, private"
    )


class Constraints(BaseModel):
    """Design and code compliance constraints"""
    code_requirements: List[str] = Field(
        default_factory=list,
        description="Building code requirements"
    )
    design_notes: Optional[str] = Field(None, description="Design guidelines")
    max_occupancy: Optional[int] = Field(None, ge=1, description="Maximum occupancy")
    accessibility: bool = Field(False, description="ADA/accessibility compliant")


class Metadata(BaseModel):
    """Plan metadata and version tracking"""
    version: str = Field("1.0", description="Schema version")
    created: str = Field(..., description="Creation timestamp (ISO 8601)")
    modified: str = Field(..., description="Last modified timestamp (ISO 8601)")
    created_by: str = Field(..., description="Creator identifier")
    source: Optional[str] = Field(None, description="Source file or system")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")


class Building(BaseModel):
    """Building-level information"""
    name: str = Field(..., description="Building name")
    address: Optional[str] = Field(None, description="Building address")
    total_area: float = Field(..., gt=0, description="Total building area (sqm)")
    floors: int = Field(1, ge=1, description="Number of floors")
    area_unit: str = Field("sqm", description="Area measurement unit")


class FloorPlan(BaseModel):
    """Complete floor plan specification"""
    id: str = Field(..., description="Unique floor plan identifier")
    metadata: Metadata = Field(..., description="Plan metadata")
    building: Optional[Building] = Field(None, description="Building information")
    
    # Floor plan details
    name: str = Field(..., description="Floor plan name")
    level: int = Field(0, description="Floor level number")
    total_area: float = Field(..., gt=0, description="Total floor plan area (sqm)")
    
    # Spatial elements
    rooms: List[Room] = Field(..., min_items=1, description="List of rooms")
    adjacencies: List[Adjacency] = Field(
        default_factory=list,
        description="Room relationships"
    )
    zones: List[Zone] = Field(default_factory=list, description="Functional zones")
    
    # Constraints and compliance
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Design constraints"
    )
    
    @validator('rooms')
    def validate_unique_room_ids(cls, v):
        """Ensure all room IDs are unique"""
        ids = [room.id for room in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Room IDs must be unique")
        return v
    
    @validator('total_area')
    def validate_total_area(cls, v, values):
        """Check if total area matches sum of room areas"""
        if 'rooms' in values:
            room_area_sum = sum(room.area for room in values['rooms'])
            # Allow 10% tolerance for walls and common areas
            if v < room_area_sum * 0.9 or v > room_area_sum * 1.3:
                raise ValueError(
                    f"Total area {v} inconsistent with room areas {room_area_sum}"
                )
        return v
    
    @property
    def bedroom_count(self) -> int:
        """Count number of bedrooms"""
        return sum(1 for room in self.rooms if room.type == RoomType.BEDROOM)
    
    @property
    def bathroom_count(self) -> int:
        """Count number of bathrooms"""
        return sum(1 for room in self.rooms if room.type == RoomType.BATHROOM)
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Retrieve room by ID"""
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return self.model_dump()
    
    def to_search_text(self) -> str:
        """Generate searchable text representation"""
        text_parts = [
            f"Name: {self.name}",
            f"Total area: {self.total_area} square meters",
            f"Bedrooms: {self.bedroom_count}",
            f"Bathrooms: {self.bathroom_count}",
            f"Rooms: {', '.join(room.type.value for room in self.rooms)}",
        ]
        
        if self.zones:
            text_parts.append(
                f"Zones: {', '.join(zone.name for zone in self.zones)}"
            )
        
        if self.constraints.design_notes:
            text_parts.append(f"Notes: {self.constraints.design_notes}")
        
        return " | ".join(text_parts)


# Example floor plan for testing
EXAMPLE_FLOOR_PLAN = {
    "id": "plan_001",
    "metadata": {
        "version": "1.0",
        "created": "2025-01-15T10:00:00Z",
        "modified": "2025-01-15T10:00:00Z",
        "created_by": "system",
        "source": "manual_input",
        "tags": ["residential", "2bhk", "modern"]
    },
    "building": {
        "name": "Residential Tower Block A",
        "address": "123 Main St, City",
        "total_area": 45000,
        "floors": 15,
        "area_unit": "sqm"
    },
    "name": "2BHK Residential Unit - Type A",
    "level": 5,
    "total_area": 95.0,
    "rooms": [
        {
            "id": "room_001",
            "type": "bedroom",
            "name": "Master Bedroom",
            "area": 18.0,
            "dimensions": {
                "length": 4.5,
                "width": 4.0,
                "unit": "m"
            },
            "features": {
                "windows": 2,
                "doors": 1,
                "ensuite": True,
                "walk_in_closet": True
            },
            "geometry": {
                "vertices": [[0, 0], [4.5, 0], [4.5, 4.0], [0, 4.0]],
                "coordinates": "cartesian"
            },
            "floor_level": 0
        },
        {
            "id": "room_002",
            "type": "bedroom",
            "name": "Bedroom 2",
            "area": 12.0,
            "dimensions": {
                "length": 4.0,
                "width": 3.0,
                "unit": "m"
            },
            "features": {
                "windows": 1,
                "doors": 1
            },
            "floor_level": 0
        },
        {
            "id": "room_003",
            "type": "bathroom",
            "name": "Master Bath",
            "area": 6.0,
            "dimensions": {
                "length": 3.0,
                "width": 2.0,
                "unit": "m"
            },
            "features": {
                "windows": 1,
                "doors": 1
            },
            "floor_level": 0
        },
        {
            "id": "room_004",
            "type": "bathroom",
            "name": "Common Bath",
            "area": 4.0,
            "dimensions": {
                "length": 2.0,
                "width": 2.0,
                "unit": "m"
            },
            "features": {
                "windows": 1,
                "doors": 1
            },
            "floor_level": 0
        },
        {
            "id": "room_005",
            "type": "kitchen",
            "area": 10.0,
            "dimensions": {
                "length": 5.0,
                "width": 2.0,
                "unit": "m"
            },
            "features": {
                "windows": 1,
                "doors": 1
            },
            "floor_level": 0
        },
        {
            "id": "room_006",
            "type": "living_room",
            "area": 25.0,
            "dimensions": {
                "length": 5.0,
                "width": 5.0,
                "unit": "m"
            },
            "features": {
                "windows": 2,
                "doors": 1,
                "balcony": True
            },
            "floor_level": 0
        },
        {
            "id": "room_007",
            "type": "balcony",
            "area": 8.0,
            "dimensions": {
                "length": 4.0,
                "width": 2.0,
                "unit": "m"
            },
            "features": {
                "doors": 1
            },
            "floor_level": 0
        }
    ],
    "adjacencies": [
        {
            "room1": "room_001",
            "room2": "room_003",
            "type": "access",
            "has_wall": False,
            "has_door": True,
            "description": "Ensuite bathroom access"
        },
        {
            "room1": "room_006",
            "room2": "room_007",
            "type": "access",
            "has_wall": False,
            "has_door": True,
            "description": "Balcony access from living room"
        }
    ],
    "zones": [
        {
            "name": "sleeping_zone",
            "rooms": ["room_001", "room_002", "room_003", "room_004"],
            "separation_level": "private"
        },
        {
            "name": "living_zone",
            "rooms": ["room_005", "room_006", "room_007"],
            "separation_level": "public"
        }
    ],
    "constraints": {
        "code_requirements": ["natural_light_minimum", "ventilation_minimum"],
        "design_notes": "Maximize cross-ventilation, minimize noise",
        "accessibility": False
    }
}


if __name__ == "__main__":
    # Validate example floor plan
    plan = FloorPlan(**EXAMPLE_FLOOR_PLAN)
    print(f"Floor plan validated: {plan.name}")
    print(f"Bedrooms: {plan.bedroom_count}, Bathrooms: {plan.bathroom_count}")
    print(f"Total area: {plan.total_area} sqm")
    print(f"\nSearchable text:\n{plan.to_search_text()}")
