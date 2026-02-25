def evaluate(expression: str) -> float:
    """
    Evaluate a mathematical expression string and return the result as a float.
    
    Supports:
    - Basic arithmetic operators: +, -, *, /
    - Parentheses (including nested)
    - Proper order of operations
    
    Args:
        expression: String containing the mathematical expression
        
    Returns:
        float: The evaluated result
    """
    # Remove whitespace
    expression = expression.replace(' ', '')
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    # Initialize parser state
    index = 0
    
    def peek():
        """Look at the current character without consuming it"""
        nonlocal index
        if index >= len(expression):
            return None
        return expression[index]
    
    def consume():
        """Consume and return the current character"""
        nonlocal index
        if index >= len(expression):
            return None
        char = expression[index]
        index += 1
        return char
    
    def parse_number():
        """Parse a number (integer or float)"""
        nonlocal index
        start = index
        
        # Handle negative numbers
        if peek() == '-':
            consume()
        
        # Parse digits and decimal point
        while peek() and (peek().isdigit() or peek() == '.'):
            consume()
        
        if start == index:
            raise ValueError(f"Expected number at position {index}")
        
        return float(expression[start:index])
    
    def parse_factor():
        """Parse a factor: number or parenthesized expression"""
        char = peek()
        
        if char is None:
            raise ValueError(f"Unexpected end of expression at position {index}")
        elif char == '(':
            consume()  # consume '('
            result = parse_expression()
            if peek() != ')':
                raise ValueError(f"Expected ')' at position {index}")
            consume()  # consume ')'
            return result
        elif char == '-':
            consume()  # consume '-'
            return -parse_factor()
        elif char.isdigit() or char == '.':
            return parse_number()
        else:
            raise ValueError(f"Unexpected character '{char}' at position {index}")
    
    def parse_term():
        """Parse a term: factor followed by * or / operations"""
        result = parse_factor()
        
        while peek() in ['*', '/']:
            op = consume()
            right = parse_factor()
            if op == '*':
                result *= right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def parse_expression():
        """Parse an expression: term followed by + or - operations"""
        result = parse_term()
        
        while peek() in ['+', '-']:
            op = consume()
            right = parse_term()
            if op == '+':
                result += right
            else:  # op == '-'
                result -= right
        
        return result
    
    # Parse the entire expression
    result = parse_expression()
    
    # Check if we consumed the entire string
    if index < len(expression):
        raise ValueError(f"Unexpected character '{expression[index]}' at position {index}")
    
    return result