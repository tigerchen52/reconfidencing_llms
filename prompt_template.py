ALL_PROMPTS = {
    "birth_date": """
    Context: Suppose you know birth dates of all people.  
    Instruction: what is the birth date of {a}?   
    Constraint: The output formating should be this pattern: yyyy-mm-dd.  
    Demonstrations:  
    (1) Question: what is the birth date of Albert Einstein?  
    Output: 1879-04-14
    (2) Question: what is the birth date of Michael Jackson?  
    Output: 1958-08-29
    """,
    "birth_place": """
    Context: Suppose you know birth places of all people.  
    Instruction: what is the birth place of {a}?    
    Demonstrations:  
    (1) Question: what is the birth place of Albert Einstein?  
    Output: Ulm
    (2) Question: what is the birth place of Michael Jackson?  
    Output: Gary
    """,
    "father":"""
    Context: Suppose you know fathers of all people.  
    Instruction: who is the father of {a}?   
    Demonstrations:  
    (1) Question: who is the father of Albert Einstein?  
    Output: Hermann Einstein
    (2) Question: who is the father of Michael Jackson?  
    Output: Joe Jackson
    """
        
}

