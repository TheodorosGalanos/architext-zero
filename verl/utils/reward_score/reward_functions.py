from shapely.geometry import Polygon
from dataclasses import dataclass
from typing import Dict, Tuple
# Scale factor: 256 units = ~64 feet
# Therefore 1 unit ≈ 0.25 feet
# Area in our system would be in square units (not square feet)

## ROOM DIMENSIONS RULES ##

MIN_ROOM_DIMENSIONS = {
    'bedroom': {
        'primary': 1200,    # ~120 sq ft -> ~30x40 units
        'secondary': 800,   # ~80 sq ft -> ~24x32 units
        'min_width': 32     # 8 feet -> 32 units
    },
    'bathroom': {
        'full': 400,        # ~40 sq ft -> ~20x20 units
        'half': 180,        # ~18 sq ft -> ~12x15 units
        'min_width': 20     # 5 feet -> 20 units
    },
    'living_room': {        # Note the underscore
        'min_area': 1500,   # ~150 sq ft -> ~40x40 units
        'min_width': 48     # 12 feet -> 48 units
    },
    'dining_room': {        # Added dining room
        'min_area': 1200,   # ~120 sq ft
        'min_width': 40     # 10 feet -> 40 units
    },
    'kitchen': {
        'min_area': 700,    # ~70 sq ft -> ~28x25 units
        'min_width': 32     # 8 feet -> 32 units
    },
    'corridor': {
        'min_width': 12,    # 3 feet -> 12 units
        'max_width': 20     # 5 feet -> 20 units
    }
}

# Define preferred and prohibited adjacencies
ADJACENCY_RULES = {
    'kitchen': {
        'preferred': ['living_room', 'dining_room'],
        'prohibited': ['bedroom', 'bathroom']
    },
    'bedroom': {
        'preferred': ['bathroom', 'corridor'],
        'prohibited': ['kitchen', 'living_room']  # Unless specifically requested
    },
    'bathroom': {
        'preferred': ['bedroom', 'corridor'],
        'prohibited': ['kitchen', 'living_room']
    }
}

@dataclass
class RewardWeights:
    """Weights for different components of the reward function"""
    format_validity: float = 0.2    # Basic format and geometric validity
    room_quality: float = 0.2       # Room dimensions and proportions
    adjacency: float = 0.2          # Spatial relationships
    efficiency: float = 0.1         # Space utilization
    constraints: float = 0.3        # Prompt constraint satisfaction

class LayoutRewardCalculator:
    def __init__(self, weights: RewardWeights = None): # type: ignore
        self.weights = weights or RewardWeights()
    
    def calculate_format_validity_reward(self, layout: Dict) -> Tuple[float, str]:
        """
        Check format validity and geometric correctness
        Returns: (reward in [0,1], message)
        """
        is_valid, message = validate_layout_format(layout)
        if not is_valid:
            return 0.0, message
            
        # Check if all polygons are valid
        try:
            for room_type, coords in layout.items():
                poly = Polygon(coords)
                if not poly.is_valid:
                    return 0.5, f"Invalid polygon for {room_type}"
        except Exception:
            return 0.0, "Error creating polygons"
            
        return 1.0, "Valid format"
    
    def calculate_room_quality_reward(self, layout: Dict) -> Tuple[float, Dict]:
        """
        Evaluate room dimensions and proportions
        Returns: (reward in [0,1], details)
        """
        try:
            if not layout:
                return 0.0, {"error": "Empty layout"}
                
            total_rooms = len(layout)
            valid_rooms = 0
            details = {}
            
            # Minimum reasonable room size in our coordinate system
            MIN_REASONABLE_AREA = 400  # 20x20 units minimum
            
            for room_type, coordinates in layout.items():
                room_score = 0.0
                area = calculate_area(coordinates)
                width, height = calculate_width_height(coordinates)
                
                # Penalize tiny rooms heavily
                if area < MIN_REASONABLE_AREA:
                    room_score = 0.1  # Minimal score for valid format but too small
                    details[room_type] = {"score": room_score, "reason": "Room too small"}
                    valid_rooms += room_score
                    continue
                
                # Check basic constraints (area, width)
                if check_room_constraints(room_type, coordinates):
                    room_score += 0.6
                
                # Check proportions
                ratio = max(width, height) / min(width, height)
                if ratio <= 2.5:
                    room_score += 0.4
                
                valid_rooms += room_score
                details[room_type] = {"score": room_score, "area": area, "ratio": ratio}
                
            # Normalize to [0,1]
            final_score = valid_rooms / (total_rooms * 1.0)
            return final_score, details
            
        except Exception as e:
            print(f"Error in room quality calculation: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def calculate_adjacency_reward(self, layout: Dict) -> Tuple[float, Dict]:
        """
        Evaluate spatial relationships
        Returns: (reward in [0,1], details)
        """
        try:
            if not layout:
                return 0.0, {"error": "Empty layout"}
                
            adjacencies = get_all_adjacencies(layout)
            total_score = 0.0
            details = {}
            
            # Score preferred adjacencies
            for room_type, adjacent_rooms in adjacencies.items():
                room_score = 0.0
                if room_type in ADJACENCY_RULES:
                    rules = ADJACENCY_RULES[room_type]
                    
                    # Preferred adjacencies (0.6 of room score)
                    preferred_count = sum(1 for adj in adjacent_rooms 
                                    if adj in rules['preferred'])
                    if preferred_count > 0:
                        room_score += 0.6 * (preferred_count / len(rules['preferred']))
                    
                    # Avoid prohibited adjacencies (0.4 of room score)
                    prohibited_count = sum(1 for adj in adjacent_rooms 
                                        if adj in rules['prohibited'])
                    if prohibited_count == 0:
                        room_score += 0.4
                        
                total_score += room_score
                details[room_type] = room_score
                
            # Normalize to [0,1]
            if len(layout) > 0:
                final_score = total_score / len(layout)
            else:
                final_score = 0.0
                
            return final_score, details
            
        except Exception as e:
            print(f"Error in adjacency calculation: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def calculate_efficiency_reward(self, layout: Dict) -> Tuple[float, Dict]:
        """
        Evaluate space utilization
        Returns: (reward in [0,1], details)
        """
        try:
            if not layout:
                return 0.0, {"error": "Empty layout"}
                
            metrics = calculate_layout_efficiency(layout)
            
            # Check for calculation errors
            if 'error' in metrics:
                return 0.0, metrics
                
            # Check if total area is too small
            MIN_TOTAL_AREA = 1200  # Minimum reasonable house size
            if metrics['total_area'] < MIN_TOTAL_AREA:
                return 0.1, {
                    "error": "Total area too small",
                    "total_area": metrics['total_area'],
                    "min_required": MIN_TOTAL_AREA
                }
            
            # Score components
            area_score = min(metrics['efficiency_ratio'], 0.9)  # Cap at 0.9 to avoid too tight packing
            gaps_score = 0.1 if not metrics['has_gaps'] else 0.0
            
            final_score = area_score + gaps_score
            return final_score, metrics
            
        except Exception as e:
            print(f"Error in efficiency calculation: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def calculate_constraint_reward(self, layout: Dict, prompt: str) -> Tuple[float, Dict]:
        """
        Evaluate prompt constraint satisfaction
        Returns: (reward in [0,1], details)
        """
        details = {}
        
        # Extract constraints
        room_constraints = extract_room_constraints(prompt)
        location_constraints = extract_rooms_from_prompt(prompt)
        
        # Score room counts/combinations (0.5 of total)
        is_valid, count_message = validate_room_constraints(layout, room_constraints) # type: ignore
        count_score = 0.5 if is_valid else 0.0
        details['room_counts'] = count_score
        
        # Score location/adjacency constraints (0.5 of total)
        if location_constraints:
            is_valid = validate_layout_against_prompt(layout, prompt)
            location_score = 0.5 if is_valid else 0.0
            details['location'] = location_score
        else:
            location_score = 0.5  # No constraints to satisfy
            details['location'] = 0.5
            
        final_score = count_score + location_score
        return final_score, details
    
    def calculate_total_reward(self, layout: Dict, prompt: str) -> Tuple[float, Dict]:
        """
        Calculate weighted combination of all rewards
        Returns: (final_reward in [0,1], detailed_breakdown)
        """
        rewards = {}
        
        # Calculate individual rewards
        format_reward, format_msg = self.calculate_format_validity_reward(layout)
        rewards['format'] = {
            'score': format_reward,
            'weight': self.weights.format_validity,
            'details': format_msg
        }
        
        # Only continue if format is valid
        if format_reward == 0:
            return 0.0, rewards
            
        quality_reward, quality_details = self.calculate_room_quality_reward(layout)
        rewards['quality'] = {
            'score': quality_reward,
            'weight': self.weights.room_quality,
            'details': quality_details
        }
        
        adjacency_reward, adj_details = self.calculate_adjacency_reward(layout)
        rewards['adjacency'] = {
            'score': adjacency_reward,
            'weight': self.weights.adjacency,
            'details': adj_details
        }
        
        efficiency_reward, eff_details = self.calculate_efficiency_reward(layout)
        rewards['efficiency'] = {
            'score': efficiency_reward,
            'weight': self.weights.efficiency,
            'details': eff_details
        }
        
        constraint_reward, const_details = self.calculate_constraint_reward(layout, prompt)
        rewards['constraints'] = {
            'score': constraint_reward,
            'weight': self.weights.constraints,
            'details': const_details
        }
        
        # Calculate weighted sum
        final_reward = sum(
            reward['score'] * reward['weight']
            for reward in rewards.values()
        )
        
        return final_reward, rewards
    

def calculate_area(coordinates):
    """Calculate area of a room from its coordinates"""
    area = 0
    coords = coordinates + [coordinates[0]]  # Close the polygon
    for i in range(len(coordinates)):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2

def calculate_width_height(coordinates):
    """Calculate width and height of room's bounding box"""
    x_min, y_min, x_max, y_max = calculate_bounding_box(coordinates)
    return x_max - x_min, y_max - y_min


def calculate_centroid(coordinates):
    """Calculate the centroid of a room"""
    x_coords = [x for x, _ in coordinates]
    y_coords = [y for _, y in coordinates]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def check_room_constraints(room_type, coordinates):
    """
    Check room constraints considering non-convex shapes
    """
    area = calculate_area(coordinates)
    width, height = calculate_width_height(coordinates)
    min_dimension = min(width, height)
    
    # Check minimum area
    if room_type in MIN_ROOM_DIMENSIONS:
        if 'min_area' in MIN_ROOM_DIMENSIONS[room_type]:
            if area < MIN_ROOM_DIMENSIONS[room_type]['min_area']:
                return False
    
    # Check minimum width (using bounding box)
    if room_type in MIN_ROOM_DIMENSIONS:
        if 'min_width' in MIN_ROOM_DIMENSIONS[room_type]:
            if min_dimension < MIN_ROOM_DIMENSIONS[room_type]['min_width']:
                return False
    
    # Special check for corridors (maximum width)
    if room_type == 'corridor':
        if width > MIN_ROOM_DIMENSIONS['corridor']['max_width']:
            return False
    
    # For non-convex shapes, we might want to check the ratio between
    # actual area and bounding box area to ensure it's not too irregular
    bbox_area = width * height
    if area / bbox_area < 0.4:  # Allow L-shapes, T-shapes but not too irregular
        return False
    
    return True

def parse_layout_string(layout_string: str) -> Dict:
    """Parse layout string into a dictionary of room types and coordinates."""
    try:
        layout_dict = {}
        room_counters = {}
        
        # Basic validation
        if not layout_string or not isinstance(layout_string, str):
            return {}
            
        # Split into individual room definitions
        room_definitions = layout_string.split(', ')
        if not room_definitions:
            return {}
            
        for room_def in room_definitions:
            if not ': ' in room_def:
                return {}
                
            room_type, coords_str = room_def.split(': ')
            room_type = room_type.strip()
            
            # Parse coordinates
            coords_list = []
            coords_pairs = coords_str.replace(')(', '|').strip('()').split('|')
            
            try:
                for coord_pair in coords_pairs:
                    x, y = map(int, coord_pair.split(','))
                    if not (0 <= x <= 256 and 0 <= y <= 256):
                        return {}
                    coords_list.append((x, y))
            except:
                return {}
            
            # Validate minimum coordinates
            if len(coords_list) < 3:
                return {}
                
            # Handle multiple rooms of same type
            if room_type in room_counters:
                room_counters[room_type] += 1
                room_type = f"{room_type}_{room_counters[room_type]}"
            else:
                room_counters[room_type] = 1
            
            layout_dict[room_type] = coords_list
            
        return layout_dict
        
    except Exception as e:
        print(f"Error parsing layout: {str(e)}")
        return {}

def get_base_room_type(room_type):
    """
    Get base room type handling compound types like 'living_room'
    """
    # List of known compound room types
    COMPOUND_TYPES = {
        'living_room': 'living_room',
        'dining_room': 'dining_room'
    }
    
    # First check if it's a numbered room type
    if '_' in room_type and not any(compound in room_type for compound in COMPOUND_TYPES):
        base_type = room_type.split('_')[0]
    else:
        # Check for compound types
        base_type = next((full_type for full_type in COMPOUND_TYPES if full_type in room_type), room_type)
    
    return base_type

def calculate_bounding_box(coordinates):
    """Calculate the bounding box of a room"""
    x_coords = [x for x, _ in coordinates]
    y_coords = [y for _, y in coordinates]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def is_near_bedroom(bathroom_coords, layout):
    """
    Check if a bathroom is within reasonable distance of any bedroom.
    Rules:
    1. At least one bedroom should be within MAX_BATHROOM_DISTANCE
    2. Either direct adjacency or connected through corridor
    """
    MAX_BATHROOM_DISTANCE = 100  # About 25 feet in our scale
    
    # Get bathroom centroid
    bathroom_centroid = calculate_centroid(bathroom_coords)
    
    # Find all bedrooms
    bedrooms = [(room_type, coords) for room_type, coords in layout.items() 
                if room_type.startswith('bedroom')]
    
    if not bedrooms:
        return False  # No bedrooms in layout
    
    for bedroom_type, bedroom_coords in bedrooms:
        bedroom_centroid = calculate_centroid(bedroom_coords)
        distance = calculate_distance(bathroom_centroid, bedroom_centroid)
        
        if distance <= MAX_BATHROOM_DISTANCE:
            # Check if directly adjacent
            if has_shared_edge(bathroom_coords, bedroom_coords):
                return True
                
            # Check if connected through corridor
            corridors = [(room_type, coords) for room_type, coords in layout.items() 
                        if room_type.startswith('corridor')]
            
            for _, corridor_coords in corridors:
                if (has_shared_edge(bathroom_coords, corridor_coords) and 
                    has_shared_edge(bedroom_coords, corridor_coords)):
                    return True
    
    return False

## ADJACENCY RULES ##

def get_edges(coordinates):
    """Convert room coordinates into a set of edges"""
    edges = set()
    # Create edges from consecutive points
    for i in range(len(coordinates)):
        p1 = coordinates[i]
        p2 = coordinates[(i + 1) % len(coordinates)]
        # Sort points to ensure consistent edge representation
        edge = tuple(sorted([p1, p2]))
        edges.add(edge)
    return edges

def has_shared_edge(room1_coords, room2_coords):
    """
    Check if two rooms share an edge, considering non-convex shapes
    """
    edges1 = get_edges(room1_coords)
    edges2 = get_edges(room2_coords)
    
    # Find common edges
    shared_edges = edges1.intersection(edges2)
    
    # Check if any shared edge is long enough to be a valid connection
    MIN_EDGE_LENGTH = 12  # Minimum 12 units for valid connection
    
    total_shared_length = 0
    for edge in shared_edges:
        (x1, y1), (x2, y2) = edge
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        total_shared_length += length
        
    return total_shared_length >= MIN_EDGE_LENGTH

def get_all_adjacencies(layout):
    """Get all room adjacencies in the layout"""
    adjacencies = {}
    room_types = list(layout.keys())
    
    for i, (room1_type, room1_coords) in enumerate(layout.items()):
        adjacencies[room1_type] = set()
        for room2_type, room2_coords in list(layout.items())[i+1:]:
            if has_shared_edge(room1_coords, room2_coords):
                adjacencies[room1_type].add(room2_type)
                if room2_type not in adjacencies:
                    adjacencies[room2_type] = set()
                adjacencies[room2_type].add(room1_type)
    
    return adjacencies

def check_adjacency_rules(layout):
    """Check if layout follows good adjacency practices"""
    reward = 0
    adjacencies = get_all_adjacencies(layout)
    
    # Define preferred and prohibited adjacencies
    ADJACENCY_RULES = {
        'kitchen': {
            'preferred': ['living_room', 'dining_room'],
            'prohibited': ['bedroom', 'bathroom']
        },
        'bedroom': {
            'preferred': ['bathroom', 'corridor'],
            'prohibited': ['kitchen', 'living_room']  # Unless specifically requested
        },
        'bathroom': {
            'preferred': ['bedroom', 'corridor'],
            'prohibited': ['kitchen', 'living_room']
        }
    }
    
    for room_type, adjacent_rooms in adjacencies.items():
        if room_type in ADJACENCY_RULES:
            rules = ADJACENCY_RULES[room_type]
            
            # Check preferred adjacencies
            for preferred in rules['preferred']:
                if preferred in adjacent_rooms:
                    reward += 0.2
                    
            # Check prohibited adjacencies
            for prohibited in rules['prohibited']:
                if prohibited in adjacent_rooms:
                    reward -= 0.5
    
    return reward

def extract_rooms_from_prompt(prompt):
    """
    Extract room relationships from natural language prompts.
    Returns tuple of (room1, room2, relationship_type)
    """
    # Lowercase for consistent processing
    prompt = prompt.lower()
    
    # List of valid room types
    VALID_ROOMS = {
        'bedroom', 'bathroom', 'living_room', 'kitchen', 'dining_room', 
        'corridor', 'hallway'
    }
    
    # Handle common variations
    prompt = prompt.replace('living room', 'living_room')
    prompt = prompt.replace('dining room', 'dining_room')
    
    # Pattern matching for different types of adjacency phrases
    if 'adjacent to' in prompt or 'next to' in prompt:
        words = prompt.split()
        relationship = 'adjacent'
        negation = 'not' in prompt
        
        # Find the room types in the prompt
        found_rooms = []
        for word in words:
            if word in VALID_ROOMS:
                found_rooms.append(word)
        
        if len(found_rooms) == 2:
            return {
                'room1': found_rooms[0],
                'room2': found_rooms[1],
                'relationship': relationship,
                'negation': negation
            }
            
    # Handle location-based relationships
    DIRECTIONS = {'north', 'south', 'east', 'west'}
    for direction in DIRECTIONS:
        if direction in prompt:
            words = prompt.split()
            for word in words:
                if word in VALID_ROOMS:
                    return {
                        'room1': word,
                        'room2': None,
                        'relationship': 'location',
                        'direction': direction
                    }
    
    return None

## LOCATION RULES ##
def validate_layout_against_prompt(layout, prompt):
    """
    Validate if layout satisfies the constraints specified in the prompt
    """
    constraint = extract_rooms_from_prompt(prompt)
    if not constraint:
        return True  # No specific constraint to check
    
    if constraint['relationship'] == 'adjacent':
        adjacencies = get_all_adjacencies(layout)
        room1, room2 = constraint['room1'], constraint['room2']
        
        is_adjacent = room2 in adjacencies.get(room1, set())
        should_be_adjacent = not constraint['negation']
        
        return is_adjacent == should_be_adjacent
        
    elif constraint['relationship'] == 'location':
        room = constraint['room1']
        direction = constraint['direction']
        
        # Get room coordinates
        if room not in layout:
            return False
            
        room_coords = layout[room]
        
        # Calculate room centroid
        x_coords = [x for x, _ in room_coords]
        y_coords = [y for _, y in room_coords]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        
        # Check if room is in the correct direction relative to layout center
        CENTER_X, CENTER_Y = 128, 128  # Center of 256x256 grid
        
        if direction == 'west':
            return centroid_x < CENTER_X
        elif direction == 'east':
            return centroid_x > CENTER_X
        elif direction == 'north':
            return centroid_y < CENTER_Y
        elif direction == 'south':
            return centroid_y > CENTER_Y
            
    return False

## ROOM NUMBER RULES ##
def extract_room_constraints(prompt):
    """
    Extract room count constraints from prompts like:
    - "a house with three bedrooms, two bathrooms and a corridor"
    - "a house with five rooms"
    """
    prompt = prompt.lower()
    
    # Number words to digits mapping
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Valid room types and their variations
    ROOM_TYPES = {
        'bedroom': ['bedroom', 'bedrooms'],
        'bathroom': ['bathroom', 'bathrooms'],
        'living_room': ['living room', 'living_room'],
        'kitchen': ['kitchen', 'kitchens'],
        'corridor': ['corridor', 'corridors', 'hallway', 'hallways'],
        'dining_room': ['dining room', 'dining_room']
    }

    constraints = {
        'total_rooms': None,
        'specific_rooms': {},
        'constraint_type': None
    }

    # First, check if it's a simple total rooms constraint
    # This should only match if it's JUST about total rooms
    if 'with' in prompt and 'room' in prompt:
        words = prompt.split()
        for i, word in enumerate(words):
            if word in NUMBER_WORDS or word.isdigit():
                number = int(word) if word.isdigit() else NUMBER_WORDS[word]
                if i + 1 < len(words) and 'room' in words[i + 1]:
                    # Only set as total if there are no specific room types mentioned
                    if not any(variation in prompt 
                             for variations in ROOM_TYPES.values() 
                             for variation in variations):
                        constraints['total_rooms'] = number
                        constraints['constraint_type'] = 'total'
                        return constraints

    # If we get here, check for specific room counts
    constraints['constraint_type'] = 'specific'
    
    # Split the prompt into parts (handling 'and' separately)
    parts = prompt.replace(' and ', ', ').split(', ')
    
    for part in parts:
        words = part.split()
        for room_type, variations in ROOM_TYPES.items():
            for variation in variations:
                # Check for numbered rooms
                found_number = None
                for i, word in enumerate(words):
                    if word in NUMBER_WORDS:
                        found_number = NUMBER_WORDS[word]
                    elif word.isdigit():
                        found_number = int(word)
                    
                    if found_number and any(var in ' '.join(words[i:]) for var in variations):
                        constraints['specific_rooms'][room_type] = found_number
                        break
                
                # Check for singular forms without number (assumed to be 1)
                if variation.rstrip('s') in part and room_type not in constraints['specific_rooms']:
                    if not any(str(num) in part or word in part 
                             for num in range(2, 11) 
                             for word in list(NUMBER_WORDS.keys())[1:]):
                        constraints['specific_rooms'][room_type] = 1

    return constraints

def validate_room_constraints(layout, constraints):
    """
    Validate if layout satisfies the room constraints
    Returns (is_valid, details) tuple
    """
    if not constraints:
        return True, "No constraints to check"

    # Count rooms in layout
    room_counts = {}
    for room_type in layout:
        base_type = room_type.split('_')[0]  # Handle numbered rooms like 'bedroom_1'
        room_counts[base_type] = room_counts.get(base_type, 0) + 1

    # Check total rooms constraint
    if constraints['constraint_type'] == 'total':
        total_rooms = len(layout)
        if total_rooms != constraints['total_rooms']:
            return False, f"Expected {constraints['total_rooms']} total rooms, got {total_rooms}"
        return True, "Total room count matches"

    # Check specific room constraints
    if constraints['constraint_type'] == 'specific':
        for room_type, expected_count in constraints['specific_rooms'].items():
            actual_count = room_counts.get(room_type, 0)
            if actual_count != expected_count:
                return False, f"Expected {expected_count} {room_type}(s), got {actual_count}"
        return True, "All room counts match"

def calculate_room_constraint_reward(layout, prompt):
    """
    Calculate reward based on room constraints
    """
    constraints = extract_room_constraints(prompt)
    is_valid, details = validate_room_constraints(layout, constraints) # type: ignore
    
    if not constraints:
        return 0  # No constraints to check
    
    if is_valid:
        return 1.0
    
    # Calculate partial rewards for specific room constraints
    if constraints['constraint_type'] == 'specific':
        room_counts = {}
        for room_type in layout:
            base_type = room_type.split('_')[0]
            room_counts[base_type] = room_counts.get(base_type, 0) + 1
        
        total_constraints = len(constraints['specific_rooms'])
        matched_constraints = sum(
            1 for room_type, expected_count in constraints['specific_rooms'].items()
            if room_counts.get(room_type, 0) == expected_count
        )
        
        if matched_constraints > 0:
            return (matched_constraints / total_constraints) - 0.5
    
    return -1.0

## FORMAT VALIDATOR RULES ##

def validate_layout_format(layout):
    """
    Comprehensive format validator checking:
    1. Valid room types
    2. Valid polygon geometry
    3. Boundary constraints
    4. No overlaps between rooms
    """
    # Valid room types
    VALID_BASE_TYPES = {
        'bedroom', 'bathroom', 'living_room', 'kitchen', 
        'dining_room', 'corridor', 'hallway'
    }
    
    try:
        # Check for empty layout
        if not layout:
            return False, "Empty layout"
    
        # Check each room
        for room_type, coordinates in layout.items():
            # Check room type
            base_type = get_base_room_type(room_type)
            if base_type not in VALID_BASE_TYPES:
                return False, f"Invalid room type: {room_type} [base_type: {base_type}]"
            
            # Check minimum coordinates
            if len(coordinates) < 3:
                return False, f"Room {room_type} has fewer than 3 coordinates"
            
            # Check boundary constraints
            for x, y in coordinates:
                if not (0 <= x <= 256 and 0 <= y <= 256):
                    return False, f"Room {room_type} has coordinates outside 256x256 boundary"
            
            # Check if coordinates form a valid polygon
            try:
                poly = Polygon(coordinates)
                if not poly.is_valid:
                    return False, f"Room {room_type} coordinates don't form a valid polygon"
            except Exception as e:
                return False, f"Error creating polygon for {room_type}: {str(e)}"
        
        # Check for overlaps between rooms
        rooms = list(layout.items())
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room1_type, room1_coords = rooms[i]
                room2_type, room2_coords = rooms[j]
                try:
                    if check_room_overlap(room1_coords, room2_coords):
                        return False, f"Overlap detected between {room1_type} and {room2_type}"
                except Exception as e:
                    return False, f"Error checking overlap between {room1_type} and {room2_type}: {str(e)}"
        
        return True, "Layout format is valid"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

## 2. Boundary Checker ##
def check_room_overlap(coords1, coords2):
    """
    Check if two rooms overlap using a more accurate polygon intersection check.
    Simple bounding box check isn't sufficient as rooms might share edges without overlapping.
    """
    from shapely.geometry import Polygon
    
    # Create polygons from coordinates
    poly1 = Polygon(coords1)
    poly2 = Polygon(coords2)
    
    # Check if polygons intersect
    if poly1.intersects(poly2):
        # If they intersect, check if it's just at the edges
        intersection = poly1.intersection(poly2)
        
        # If intersection is just a line (shared edge) or points (shared vertices),
        # then it's not a real overlap
        if intersection.geom_type in ['LineString', 'MultiLineString', 'Point', 'MultiPoint']:
            return False
        return True
    
    return False

## 3. Space Efficiency Checker ##
def calculate_layout_efficiency(layout):
    """
    Calculate layout efficiency considering non-convex shapes
    """
    try:
        if not layout:
            return {
                'efficiency_ratio': 0.0,
                'has_gaps': True,
                'total_area': 0.0,
                'bbox_area': 0.0,
                'error': 'Empty layout'
            }
            
        # Calculate total used area
        total_area = sum(calculate_area(coords) for coords in layout.values())
    
        # Calculate bounding box of entire layout
        all_coords = [coord for coords in layout.values() for coord in coords]
        if not all_coords:
            return {
                'efficiency_ratio': 0.0,
                'has_gaps': True,
                'total_area': 0.0,
                'bbox_area': 0.0,
                'error': 'No coordinates found'
            }
    
        x_min = min(x for x, _ in all_coords)
        y_min = min(y for _, y in all_coords)
        x_max = max(x for x, _ in all_coords)
        y_max = max(y for _, y in all_coords)
        
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        # Calculate efficiency ratio (0 to 1)
        efficiency_ratio = total_area / bbox_area if bbox_area > 0 else 0
        
        # For non-convex shapes, we might want to be more lenient with the efficiency threshold
        has_gaps = efficiency_ratio < 0.6  # Adjusted threshold for non-convex shapes
    
        return {
            'efficiency_ratio': efficiency_ratio,
            'has_gaps': has_gaps,
            'total_area': total_area,
            'bbox_area': bbox_area
        }
        
    except Exception as e:
        print(f"Error calculating layout efficiency: {str(e)}")
        return {
            'efficiency_ratio': 0.0,
            'has_gaps': True,
            'total_area': 0.0,
            'bbox_area': 0.0,
            'error': str(e)
        }
