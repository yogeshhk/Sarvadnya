"""
Architectural Copilot - AI assistant for floor plan generation and modification
Converts natural language commands to floor plan operations
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()


class ConstraintExtractor:
    """Extract architectural constraints from natural language"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize constraint extractor"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def extract_constraints(self, description: str) -> Dict:
        """
        Extract structured constraints from natural language
        
        Args:
            description: Natural language description
            
        Returns:
            Dict with rooms, adjacencies, and constraints
        """
        system_prompt = """You are an architectural constraint extractor.
Convert natural language descriptions into structured floor plan constraints.

Return JSON with:
- rooms: List of {type, count, min_area, features}
- adjacencies: List of {room1_type, room2_type, relationship}
- constraints: {total_area, max_area, special_requirements}
- zones: List of {name, room_types}

Example:
Input: "3 bedroom apartment with ensuite master, open kitchen-living area, 1500 sq ft"
Output: {
  "rooms": [
    {"type": "bedroom", "count": 3, "min_area": 12, "features": []},
    {"type": "bedroom", "count": 1, "min_area": 15, "features": ["ensuite"], "name": "master"},
    {"type": "bathroom", "count": 3, "min_area": 4, "features": []},
    {"type": "kitchen", "count": 1, "min_area": 10, "features": ["open_concept"]},
    {"type": "living_room", "count": 1, "min_area": 20, "features": ["open_concept"]}
  ],
  "adjacencies": [
    {"room1": "master_bedroom", "room2": "master_bathroom", "type": "ensuite"},
    {"room1": "kitchen", "room2": "living_room", "type": "open"}
  ],
  "constraints": {
    "total_area": 139.35,
    "max_area": 150,
    "special_requirements": ["natural_light", "cross_ventilation"]
  },
  "zones": [
    {"name": "living_zone", "room_types": ["kitchen", "living_room"]},
    {"name": "sleeping_zone", "room_types": ["bedroom", "bathroom"]}
  ]
}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": description}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            constraints = json.loads(response.choices[0].message.content)
            return constraints
        
        except Exception as e:
            print(f"Error extracting constraints: {e}")
            return self._fallback_extract(description)
    
    def _fallback_extract(self, description: str) -> Dict:
        """Simple rule-based extraction as fallback"""
        import re
        
        desc_lower = description.lower()
        rooms = []
        constraints = {}
        
        # Extract bedroom count
        bedroom_match = re.search(r'(\d+)\s*bed', desc_lower)
        if bedroom_match:
            count = int(bedroom_match.group(1))
            rooms.append({
                "type": "bedroom",
                "count": count,
                "min_area": 12,
                "features": []
            })
        
        # Extract bathroom count
        bathroom_match = re.search(r'(\d+)\s*bath', desc_lower)
        if bathroom_match:
            count = int(bathroom_match.group(1))
            rooms.append({
                "type": "bathroom",
                "count": count,
                "min_area": 4,
                "features": []
            })
        
        # Extract area
        area_match = re.search(r'(\d+)\s*(sq\s*ft|sqft)', desc_lower)
        if area_match:
            area_sqft = int(area_match.group(1))
            constraints['total_area'] = area_sqft * 0.092903
        
        # Add standard rooms
        if 'kitchen' in desc_lower:
            rooms.append({
                "type": "kitchen",
                "count": 1,
                "min_area": 10,
                "features": []
            })
        
        if 'living' in desc_lower:
            rooms.append({
                "type": "living_room",
                "count": 1,
                "min_area": 20,
                "features": []
            })
        
        return {
            "rooms": rooms,
            "adjacencies": [],
            "constraints": constraints,
            "zones": []
        }


class FloorPlanGenerator:
    """Generate floor plans from constraints"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize floor plan generator"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=self.api_key)
        self.constraint_extractor = ConstraintExtractor(api_key)
    
    def generate(
        self, 
        description: str,
        style: str = "modern"
    ) -> Dict:
        """
        Generate floor plan from description
        
        Args:
            description: Natural language description
            style: Design style preference
            
        Returns:
            Generated floor plan dict
        """
        # Extract constraints
        constraints = self.constraint_extractor.extract_constraints(description)
        
        # Generate layout using constraints
        layout = self._generate_layout(constraints, style)
        
        # Create floor plan JSON
        floor_plan = self._create_floor_plan_json(layout, constraints)
        
        return floor_plan
    
    def _generate_layout(
        self, 
        constraints: Dict,
        style: str
    ) -> Dict:
        """
        Generate room layout from constraints
        
        In a production system, this would use:
        - Graph Neural Networks (GNN) for topology
        - HouseDiffusion or House-GAN++ for geometry
        
        For now, we use a simple algorithm
        """
        rooms = constraints.get('rooms', [])
        total_area = constraints.get('constraints', {}).get('total_area', 100)
        
        # Simple layout algorithm
        layout_rooms = []
        current_area = 0
        room_id_counter = 1
        
        for room_spec in rooms:
            room_type = room_spec.get('type')
            count = room_spec.get('count', 1)
            min_area = room_spec.get('min_area', 10)
            features = room_spec.get('features', [])
            
            for i in range(count):
                # Calculate room dimensions
                area = min_area * 1.2  # Add 20% buffer
                
                # Simple rectangular rooms
                if room_type == 'bedroom':
                    length = 4.5
                    width = area / length
                elif room_type == 'bathroom':
                    length = 2.5
                    width = area / length
                elif room_type == 'kitchen':
                    length = 5.0
                    width = area / length
                elif room_type == 'living_room':
                    length = 5.0
                    width = area / length
                else:
                    length = (area ** 0.5)
                    width = length
                
                room = {
                    'id': f'room_{room_id_counter:03d}',
                    'type': room_type,
                    'area': length * width,
                    'dimensions': {
                        'length': round(length, 2),
                        'width': round(width, 2),
                        'unit': 'm'
                    },
                    'features': self._generate_features(room_type, features),
                    'floor_level': 0
                }
                
                layout_rooms.append(room)
                current_area += room['area']
                room_id_counter += 1
        
        return {
            'rooms': layout_rooms,
            'total_area': current_area,
            'adjacencies': self._generate_adjacencies(
                layout_rooms, 
                constraints.get('adjacencies', [])
            )
        }
    
    def _generate_features(self, room_type: str, required_features: List[str]) -> Dict:
        """Generate room features based on type"""
        features = {
            'windows': 1,
            'doors': 1,
            'balcony': False,
            'ensuite': False,
            'walk_in_closet': False,
            'natural_light': True
        }
        
        # Type-specific features
        if room_type == 'bedroom':
            features['windows'] = 2
        elif room_type == 'living_room':
            features['windows'] = 2
        elif room_type == 'bathroom':
            features['windows'] = 1
        elif room_type == 'kitchen':
            features['windows'] = 1
        
        # Apply required features
        for feature in required_features:
            if feature in features:
                features[feature] = True
        
        return features
    
    def _generate_adjacencies(
        self, 
        rooms: List[Dict],
        constraints: List[Dict]
    ) -> List[Dict]:
        """Generate room adjacencies based on constraints"""
        adjacencies = []
        
        # Apply constraint-based adjacencies
        for constraint in constraints:
            room1_type = constraint.get('room1')
            room2_type = constraint.get('room2')
            rel_type = constraint.get('type', 'adjacent')
            
            # Find matching rooms
            rooms1 = [r for r in rooms if r['type'] == room1_type]
            rooms2 = [r for r in rooms if r['type'] == room2_type]
            
            if rooms1 and rooms2:
                adjacencies.append({
                    'room1': rooms1[0]['id'],
                    'room2': rooms2[0]['id'],
                    'type': rel_type,
                    'has_wall': rel_type != 'open',
                    'has_door': True
                })
        
        return adjacencies
    
    def _create_floor_plan_json(
        self, 
        layout: Dict,
        constraints: Dict
    ) -> Dict:
        """Create complete floor plan JSON"""
        now = datetime.utcnow().isoformat() + 'Z'
        
        floor_plan = {
            'id': f'plan_{int(datetime.utcnow().timestamp())}',
            'metadata': {
                'version': '1.0',
                'created': now,
                'modified': now,
                'created_by': 'ai_copilot',
                'source': 'generated',
                'tags': ['ai_generated', 'residential']
            },
            'name': 'AI Generated Floor Plan',
            'level': 0,
            'total_area': round(layout['total_area'], 2),
            'rooms': layout['rooms'],
            'adjacencies': layout.get('adjacencies', []),
            'zones': constraints.get('zones', []),
            'constraints': {
                'code_requirements': ['natural_light_minimum', 'ventilation_minimum'],
                'design_notes': 'AI generated layout',
                'accessibility': False
            }
        }
        
        return floor_plan


class FloorPlanModifier:
    """Modify existing floor plans based on commands"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize floor plan modifier"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def modify(
        self, 
        floor_plan: Dict,
        command: str
    ) -> Dict:
        """
        Modify floor plan based on command
        
        Args:
            floor_plan: Existing floor plan dict
            command: Natural language modification command
            
        Returns:
            Modified floor plan dict
        """
        # Parse modification command
        modification = self._parse_modification(command, floor_plan)
        
        # Apply modification
        modified_plan = self._apply_modification(floor_plan, modification)
        
        # Update metadata
        modified_plan['metadata']['modified'] = datetime.utcnow().isoformat() + 'Z'
        
        return modified_plan
    
    def _parse_modification(self, command: str, floor_plan: Dict) -> Dict:
        """Parse modification command into structured operation"""
        system_prompt = """You are a floor plan modification parser.
Extract the modification operation from natural language commands.

Return JSON with:
- operation: "add_room", "remove_room", "modify_room", "add_feature", "remove_feature"
- target: room_id or room_type to modify
- parameters: specific changes to make

Current floor plan rooms: {rooms}

Example commands:
"Add a window to the master bedroom" → {{"operation": "add_feature", "target": "bedroom", "parameters": {{"feature": "window", "count": 1}}}}
"Make the kitchen larger" → {{"operation": "modify_room", "target": "kitchen", "parameters": {{"area_change": 1.2}}}}
"Remove the balcony" → {{"operation": "remove_room", "target": "balcony"}}
"""
        
        rooms_summary = [
            f"{r['type']} ({r['id']})" 
            for r in floor_plan.get('rooms', [])
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt.format(rooms=', '.join(rooms_summary))
                    },
                    {"role": "user", "content": command}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            modification = json.loads(response.choices[0].message.content)
            return modification
        
        except Exception as e:
            print(f"Error parsing modification: {e}")
            return {'operation': 'unknown', 'target': None, 'parameters': {}}
    
    def _apply_modification(
        self, 
        floor_plan: Dict,
        modification: Dict
    ) -> Dict:
        """Apply modification to floor plan"""
        import copy
        modified = copy.deepcopy(floor_plan)
        
        operation = modification.get('operation')
        target = modification.get('target')
        params = modification.get('parameters', {})
        
        if operation == 'add_feature':
            modified = self._add_feature(modified, target, params)
        elif operation == 'modify_room':
            modified = self._modify_room(modified, target, params)
        elif operation == 'remove_room':
            modified = self._remove_room(modified, target)
        elif operation == 'add_room':
            modified = self._add_room(modified, target, params)
        
        return modified
    
    def _add_feature(self, plan: Dict, room_target: str, params: Dict) -> Dict:
        """Add feature to a room"""
        feature = params.get('feature')
        
        for room in plan['rooms']:
            if room['type'] == room_target or room['id'] == room_target:
                if feature == 'window':
                    room['features']['windows'] += params.get('count', 1)
                elif feature in room['features']:
                    room['features'][feature] = True
        
        return plan
    
    def _modify_room(self, plan: Dict, room_target: str, params: Dict) -> Dict:
        """Modify room dimensions or area"""
        area_change = params.get('area_change', 1.0)
        
        for room in plan['rooms']:
            if room['type'] == room_target or room['id'] == room_target:
                room['area'] *= area_change
                room['dimensions']['length'] *= (area_change ** 0.5)
                room['dimensions']['width'] *= (area_change ** 0.5)
        
        # Recalculate total area
        plan['total_area'] = sum(r['area'] for r in plan['rooms'])
        
        return plan
    
    def _remove_room(self, plan: Dict, room_target: str) -> Dict:
        """Remove a room from the plan"""
        plan['rooms'] = [
            r for r in plan['rooms']
            if r['type'] != room_target and r['id'] != room_target
        ]
        
        plan['total_area'] = sum(r['area'] for r in plan['rooms'])
        return plan
    
    def _add_room(self, plan: Dict, room_type: str, params: Dict) -> Dict:
        """Add a new room to the plan"""
        area = params.get('area', 10)
        length = area ** 0.5
        
        new_room = {
            'id': f'room_new_{len(plan["rooms"]) + 1:03d}',
            'type': room_type,
            'area': area,
            'dimensions': {
                'length': round(length, 2),
                'width': round(length, 2),
                'unit': 'm'
            },
            'features': {
                'windows': 1,
                'doors': 1,
                'balcony': False,
                'ensuite': False,
                'walk_in_closet': False,
                'natural_light': True
            },
            'floor_level': 0
        }
        
        plan['rooms'].append(new_room)
        plan['total_area'] += area
        
        return plan


class ArchitecturalCopilot:
    """Main copilot interface combining generation and modification"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize architectural copilot"""
        self.generator = FloorPlanGenerator(api_key)
        self.modifier = FloorPlanModifier(api_key)
    
    def process_command(
        self, 
        command: str,
        current_plan: Optional[Dict] = None
    ) -> Dict:
        """
        Process natural language command
        
        Args:
            command: Natural language command
            current_plan: Optional existing floor plan
            
        Returns:
            Response dict with result and plan
        """
        command_lower = command.lower()
        
        # Determine if this is generation or modification
        if current_plan is None or any(
            keyword in command_lower 
            for keyword in ['create', 'generate', 'design', 'new']
        ):
            # Generate new floor plan
            plan = self.generator.generate(command)
            action = 'generated'
        else:
            # Modify existing floor plan
            plan = self.modifier.modify(current_plan, command)
            action = 'modified'
        
        return {
            'action': action,
            'command': command,
            'floor_plan': plan,
            'summary': self._generate_summary(plan)
        }
    
    def _generate_summary(self, plan: Dict) -> str:
        """Generate human-readable summary of floor plan"""
        rooms = plan.get('rooms', [])
        bedroom_count = sum(1 for r in rooms if r['type'] == 'bedroom')
        bathroom_count = sum(1 for r in rooms if r['type'] == 'bathroom')
        
        summary = f"{plan['name']}: "
        summary += f"{bedroom_count} bedroom{'s' if bedroom_count != 1 else ''}, "
        summary += f"{bathroom_count} bathroom{'s' if bathroom_count != 1 else ''}, "
        summary += f"Total area: {plan['total_area']:.1f} sqm"
        
        return summary


# Example usage
if __name__ == "__main__":
    copilot = ArchitecturalCopilot()
    
    # Test generation
    print("=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    result = copilot.process_command(
        "Create a 3 bedroom apartment with 2 bathrooms, open kitchen, and 1500 sq ft"
    )
    
    print(f"\nAction: {result['action']}")
    print(f"Command: {result['command']}")
    print(f"Summary: {result['summary']}")
    print(f"\nFloor plan preview:")
    print(json.dumps(result['floor_plan'], indent=2)[:500] + "...")
    
    # Test modification
    print("\n" + "=" * 60)
    print("MODIFICATION TEST")
    print("=" * 60)
    
    modification_result = copilot.process_command(
        "Add a balcony to the master bedroom",
        current_plan=result['floor_plan']
    )
    
    print(f"\nAction: {modification_result['action']}")
    print(f"Command: {modification_result['command']}")
    print(f"Summary: {modification_result['summary']}")
