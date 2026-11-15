# Extracted from local modified VSI-Bench checkout. See provenance.json.
import re

def extract_categories_of_interest(doc):
    
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    doc['question_type']
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    OBJ_REL_DISTANCE_TEMPLATE = """
    Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the {category}?
    """.strip()

    OBJ_REL_DIRECTION_HARD_TEMPLATE = """
    If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?
    The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).
    """.strip()

    OBJ_REL_DIRECTION_MEDIUM_TEMPLATE = """
    If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?
    An object is to my back if I would have to turn at least 135 degrees in order to face it.
    """.strip()

    OBJ_REL_DIRECTION_EASY_TEMPLATE = """
    If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
    """.strip()
    # match = None
    question_type = doc['question_type']
    question = doc['question']
    
    if question_type.startswith('object_rel_direction'):
        if question_type == "object_rel_direction_hard":
            template = OBJ_REL_DIRECTION_HARD_TEMPLATE
        elif question_type == "object_rel_direction_medium":
            template = OBJ_REL_DIRECTION_MEDIUM_TEMPLATE
        elif question_type == "object_rel_direction_easy":
            template = OBJ_REL_DIRECTION_EASY_TEMPLATE
        else:
            template = OBJ_REL_DIRECTION_HARD_TEMPLATE
        pattern = re.escape(template)
        pattern = pattern.replace(r'\{positioning_object\}', r'(?P<positioning_object>.+?)')
        pattern = pattern.replace(r'\{orienting_object\}', r'(?P<orienting_object>.+?)')
        pattern = pattern.replace(r'\{querying_object\}', r'(?P<querying_object>.+?)')
        
        if question_type == "object_rel_direction_easy":
            pattern = re.compile(
                r"^If I am standing by the "
                r"(?P<positioning_object>.*?)"
                r" and facing the "
                r"(?P<orienting_object>.*?)"
                r", is the "
                r"(?P<querying_object>.*?)"
                r" to the left or the right of the "
                r"(?P=orienting_object)\?$"
            )
        match = re.match(pattern, question)
        
    elif question_type.startswith('object_rel_distance'):
        pattern = re.escape(OBJ_REL_DISTANCE_TEMPLATE)
        pattern = pattern.replace(r'\{choice_a\}', r'(?P<choice_a>.+?)')
        pattern = pattern.replace(r'\{choice_b\}', r'(?P<choice_b>.+?)')
        pattern = pattern.replace(r'\{choice_c\}', r'(?P<choice_c>.+?)')
        pattern = pattern.replace(r'\{choice_d\}', r'(?P<choice_d>.+?)')
        pattern = pattern.replace(r'\{category\}', r'(?P<category>.+?)')
        match = re.match(pattern, question)
    
    elif question_type == "object_abs_distance":
        # Pattern: "Measuring from the closest point of each object, what is the distance between the {object1} and the {object2} (in meters)?"
        pattern = re.compile(
            r"Measuring from the closest point of each object, what is the distance between the (.+?) and the (.+?) \(in meters\)\?",
            re.IGNORECASE
        )
        match = re.match(pattern, question)
        if match:
            return {"object1": match.group(1), "object2": match.group(2)}
        return None
    
    elif question_type == "object_counting":
        # Pattern: "How many {category}(s) are in this room?"
        pattern = re.compile(
            r"How many (.+?)\(s\) are in this room\?",
            re.IGNORECASE
        )
        match = re.match(pattern, question)
        if match:
            return {"category": match.group(1)}
        return None
    
    elif question_type == "object_size_estimation":
        # Pattern: "What is the length of the longest dimension (length, width, or height) of the {object}, measured in centimeters?"
        pattern = re.compile(
            r"What is the length of the longest dimension \(length, width, or height\) of the (.+?), measured in centimeters\?",
            re.IGNORECASE
        )
        match = re.match(pattern, question)
        if match:
            return {"object": match.group(1)}
        return None
    
    elif question_type == "room_size_estimation":
        # No specific objects to extract, but we might want to return something
        # For now, return empty dict - the prompt might not need categories
        pattern = re.compile(
            r"What is the size of this room \(in square meters\)\? "
            r"If multiple rooms are shown, estimate the size of the combined space\.",
            re.IGNORECASE
        )
        match = re.match(pattern, question)
        if match:
            return {}
        return {}
        # return {}
    
    elif question_type == "obj_appearance_order":
        # Pattern: "What will be the first-time appearance order of the following categories in the video: {category1}, {category2}, {category3}, {category4}?"
        pattern = re.compile(
            r"What will be the first-time appearance order of the following categories in the video: (.+?)\?",
            re.IGNORECASE
        )
        match = re.match(pattern, question)
        if match:
            categories_str = match.group(1)
            categories = [cat.strip() for cat in categories_str.split(',')]
            return {f"category_{i+1}": cat for i, cat in enumerate(categories)}
        return None
    
    elif question_type == "route_planning":
        # Match the opening sentence
        header_pattern = re.compile(
            r"You are a robot beginning at (?P<start>.+?)\. "
            r"You want to navigate to (?P<destination>.+?)\.",
            re.IGNORECASE
        )

        header_match = header_pattern.search(question)
        if not header_match:
            return None

        result = {
            "start_position": header_match.group("start").strip(),
            "destination": header_match.group("destination").strip(),
        }

        # Extract all navigation landmarks
        # Matches: "Go forward until the TV", "Go forward until the shower", etc.
        step_pattern = re.compile(
            r"Go forward until the (?P<object>.+?)(?:\.|\s*\d+\.|\s*\[|$)",
            re.IGNORECASE
        )

        steps = step_pattern.findall(question)
        if steps:
            result["intermediate_objects"] = [s.strip() for s in steps]

        return result
        
    if match is not None and hasattr(match, 'groupdict'):
        return match.groupdict()
    elif match is not None:
        # match is already processed above
        return match if isinstance(match, dict) else None
    else:
        return {}
