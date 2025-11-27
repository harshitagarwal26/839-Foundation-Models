def get_verifier_reward(text):
    """
    Rewards the model for using 'Detective' vocabulary.
    +1.0 for each unique crime/mystery keyword found.
    Max Score: 5.0
    """
    score = 0.0
    text = text.lower()
    
    # Relevant keywords found frequently in Sherlock Holmes
    CRIME_WORDS = set([
        "clue", "evidence", "blood", "murder", "case", "trace", 
        "deduction", "logic", "police", "inspector", "crime", 
        "magnifying", "footprint", "weapon", "motive", "victim",
        "watson", "holmes", "mystery", "solved", "suspect", "arrest",
        "dead", "gun", "knife", "poison", "witness"
    ])
    
    # Tokenize simply to match whole words
    import re
    words = set(re.findall(r'\w+', text))
    
    # Count how many target words were used
    match_count = 0
    for word in words:
        if word in CRIME_WORDS:
            match_count += 1
            
    # --- CHANGE IS HERE ---
    # Multiply by 5.0. 
    # 1 word = 5.0 reward. 
    # This creates a strong "Advantage" signal.
    score = float(match_count) * 5.0
    
    if score > 25.0: # Cap at 5 words
        score = 25.0
        
    return score

if __name__ == "__main__":
    # Quick Test
    print(f"Score: {get_verifier_reward('The sun was shining.')}") # 0.0
    print(f"Score: {get_verifier_reward('Holmes found a clue.')}")  # 2.0 (Holmes, clue)
    print(f"Score: {get_verifier_reward('The murder evidence was blood.')}") # 3.0