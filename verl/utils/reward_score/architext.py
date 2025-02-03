import re
import random
from .reward_functions import LayoutRewardCalculator, RewardWeights, parse_layout_string

def extract_solution(solution_str):
    """Extract the layout from the solution string."""
    try:
        if random.random() < 0.1:  # 10% chance to print
            print("\nExtraction Debug:")
            print("Original string:", repr(solution_str))
        
        # Remove everything before the first "Assistant:"
        if "Assistant:" in solution_str:
            solution_str = solution_str.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in solution_str:
            solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
            
        # Try different regex patterns
        patterns = [
            r'<layout>(.*?)</layout>',
            r'<layout>\s*(.*?)\s*</layout>',
            r'<layout>[\s\S]*?(.*?)[\s\S]*?</layout>',
            r'<layout>\n?(.*?)\n?</layout>'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, solution_str, re.DOTALL)
            print(f"\nPattern {i+1}: {pattern}")
            if match:
                print(f"Match found with pattern {i+1}")
                final_answer = match.group(1).strip()
                print("Final answer:", repr(final_answer))
                
                # Clean up any newlines and extra spaces
                final_answer = ' '.join(line.strip() for line in final_answer.split('\n'))
                print("Cleaned answer:", repr(final_answer))
                
                if final_answer:
                    return final_answer
                    
        print("No valid match found")
        return None
        
    except Exception as e:
        print(f"Error extracting solution: {str(e)}")
        return None

def extract_solution_fewshot(solution_str):
    """Extract the layout from the solution string."""
    try:
        print("\nExtraction Debug:")
        print("Original string length:", len(solution_str))
        
        # Split by all assistant markers and take the last one
        if "<|im_start|>assistant" in solution_str:
            parts = solution_str.split("<|im_start|>assistant")
            solution_str = parts[-1]  # Take the last assistant response
            print("Found last assistant response")
        elif "Assistant:" in solution_str:
            parts = solution_str.split("Assistant:")
            solution_str = parts[-1]  # Take the last assistant response
            print("Found last Assistant: response")
            
        print("Looking for pattern in last response section")
        
        # Try different regex patterns
        patterns = [
            r'<layout>(.*?)</layout>',
            r'<layout>\s*(.*?)\s*</layout>',
            r'<layout>[\s\S]*?(.*?)[\s\S]*?</layout>',
            r'<layout>\n?(.*?)\n?</layout>'
        ]
        
        # Find all matches and take the last one
        for i, pattern in enumerate(patterns):
            matches = list(re.finditer(pattern, solution_str, re.DOTALL))
            if matches:
                # Take the last match
                match = matches[-1]
                print(f"Match found with pattern {i+1} - using last match")
                final_answer = match.group(1).strip()
                print("Final answer:", repr(final_answer))
                
                # Clean up any newlines and extra spaces
                final_answer = ' '.join(line.strip() for line in final_answer.split('\n'))
                print("Cleaned answer:", repr(final_answer))
                
                if final_answer:
                    return final_answer
                    
        print("No valid match found")
        return None
        
    except Exception as e:
        print(f"Error extracting solution: {str(e)}")
        return None

def compute_score(solution_str, ground_truth, weights=RewardWeights(), debug=True, few_shot=True):
    """Compute reward score for the completion with detailed debugging."""
    try:
        if debug:
            print("\nDebug Information:")
            
        # Extract layout from completion
        if(few_shot):
            layout_str = extract_solution_fewshot(solution_str)
        else:
            layout_str = extract_solution(solution_str)
        if debug:
            print(f"1. Extracted layout: {layout_str}")
        if layout_str is None:
            return 0.0
            
        # Parse layout string into dictionary
        layout = parse_layout_string(layout_str)
        if debug:
            print(f"2. Parsed layout: {layout}")
        if not layout:
            return 0.0

        # Ensure prompt is a string
        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get('prompt', '')  # Extract prompt string from dict
        elif prompt is None:
            prompt = ''
            
        # Calculate reward
        calculator = LayoutRewardCalculator(weights)
        final_reward = calculator.calculate_total_reward(layout, ground_truth)
        
        return final_reward
        
    except Exception as e:
        print(f"Error computing score: {str(e)}")
        return 0.0

# Example usage:
def test_format_completion():
    # Test cases
    test_completions = [
        """
        <requirements>Need three bedrooms and two bathrooms</requirements>
        <planning>Arranging bedrooms in north wing with nearby bathrooms</planning>
        <layout>
        bedroom: (187,135)(128,135)(128,77)(187,77), 
        bedroom: (172,209)(128,209)(128,150)(157,150)(157,165)(172,165), 
        living_room: (128,135)(55,135)(55,47)(128,47)
        </layout>
        """,
        
        """
        <layout>bedroom: (0,0)(1,0)(1,1)(0,1)</layout>
        """,
        
        # Invalid cases
        "<layout>invalid format</layout>",
        "no tags here",
        "<layout>bedroom: (0,0)(1,0)(1,1)</layout>"  # incomplete coordinates
    ]
    prompt = "a house with three bedrooms and two bathrooms"
    for i, completion in enumerate(test_completions):
        print(f"\nTest case {i+1}:")
        print("Input:", completion.strip())
        result = compute_score(completion, prompt, weights=RewardWeights())
        print("Output:", result)

if __name__ == "__main__":
    test_format_completion()
