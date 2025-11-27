import re

# Top common English "glue" words that appear in almost ALL valid text
# This list is general-purpose and not specific to Sherlock or Shakespeare.
COMMON_ENGLISH_WORDS = set([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 
    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 
    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 
    'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 
    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 
    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 
    'give', 'day', 'most', 'us'
])

def get_diversity_score(text):
    """
    Calculates the ratio of valid common English words.
    
    - High Score (near 1.0): Text is coherent English (contains many glue words).
    - Low Score (near 0.0): Text is gibberish or random character noise.
    """
    # Tokenize and lowercase
    words = re.findall(r'\w+', text.lower())
    
    if not words:
        return 0.0
    
    # Count how many words are valid common English
    valid_count = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    
    # Calculate Ratio
    # We normalize by length so long gibberish strings don't win by accident.
    ratio = valid_count / len(words)
    
    # SCALING:
    # Normal English is about 40-50% common words. 
    # We multiply by 2.0 to push the score closer to 1.0 for good text.
    score = ratio * 2.0
    
    # Cap at 1.0
    if score > 1.0: 
        score = 1.0
        
    return score

if __name__ == "__main__":
    print(f"Score 'The man went to the store': {get_diversity_score('The man went to the store')}") # Should be high
    print(f"Score 'weristland wipanss': {get_diversity_score('weristland wipanss')}")             # Should be 0.0