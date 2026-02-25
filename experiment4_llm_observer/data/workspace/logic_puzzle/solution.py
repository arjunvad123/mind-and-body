def solve_puzzle():
    """
    Solve the logic puzzle about three friends and their jobs.
    
    Three friends — Alice, Bob, and Carol — each have different jobs: Teacher, Doctor, and Engineer.
    
    Clues:
    1. Alice is not the Engineer.
    2. Bob is not the Teacher.
    3. The Doctor is not Alice.
    4. Carol is older than the Engineer (so Carol is not the Engineer).
    
    Returns:
        dict: A dictionary mapping each person to their job
    """
    # From clues 1 and 4: Neither Alice nor Carol can be the Engineer
    # Therefore, Bob must be the Engineer
    
    # From clue 3: Alice is not the Doctor
    # Since Bob is the Engineer, Alice must be the Teacher
    
    # By elimination, Carol must be the Doctor
    
    return {
        "Alice": "Teacher",
        "Bob": "Engineer", 
        "Carol": "Doctor"
    }