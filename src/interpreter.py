import re
import sys
from enum import Enum

class TokenType(Enum):
    PRINT = 'PRINT'
    VARIABLE = 'VARIABLE'
    ASSIGN = 'ASSIGN'
    INPUT = 'INPUT'
    IF = 'IF'
    ELSE = 'ELSE'
    ELSEIF = 'ELSEIF'
    OPERATOR = 'OPERATOR'
    STRING = 'STRING'
    NUMBER = 'NUMBER'
    IDENTIFIER = 'IDENTIFIER'
    COMMENT = 'COMMENT'
    LBRACE = 'LBRACE'
    RBRACE = 'RBRACE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    COMMA = 'COMMA'
    EOF = 'EOF'
    SEMICOLON = 'SEMICOLON'
    ECHO = 'ECHO'
    FUNC = 'FUNC'
    CUE = 'CUE'
    RETURN = 'RETURN'
    WHILE = 'WHILE'

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char and self.current_char != '\n':
            self.advance()

    def string(self):
        result = ''
        self.advance()  # Skip the opening quote
        while self.current_char and self.current_char != '"':
            result += self.current_char
            self.advance()
        self.advance()  # Skip the closing quote
        return Token(TokenType.STRING, result)

    def identifier_token(self):
        result = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        # Map the identifier to the appropriate token type
        token_type = TokenType.IDENTIFIER
        if result.lower() == 'cue':
            token_type = TokenType.CUE
        elif result.lower() == 'echo':
            token_type = TokenType.ECHO
        elif result.lower() == 'if':
            token_type = TokenType.IF
        elif result.lower() == 'else':
            token_type = TokenType.ELSE
        elif result.lower() == 'elseif':
            token_type = TokenType.ELSEIF
        elif result.lower() == 'func':
            token_type = TokenType.FUNC
        elif result.lower() == 'return':
            token_type = TokenType.RETURN
        elif result.lower() == 'while':
            token_type = TokenType.WHILE

        return Token(token_type, result)

    def number(self):
        result = ''
        while self.current_char and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        if self.current_char == '.':
            result += '.'
            self.advance()
            while self.current_char and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(TokenType.NUMBER, float(result))
        return Token(TokenType.NUMBER, int(result))

    def get_next_token(self):
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '~' and self.peek() == '~':
                self.advance()
                self.advance()
                if self.current_char == '\n':
                    self.skip_comment()
                else:
                    while self.current_char and (self.current_char != '~' or self.peek() != '~'):
                        self.advance()
                    self.advance()
                    self.advance()
                continue

            if self.current_char == '"':
                return self.string()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier_token()

            if self.current_char == '$':
                return self.variable()

            if self.current_char == '=':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '==')
                else:
                    self.advance()
                    return Token(TokenType.ASSIGN, '=')

            if self.current_char == '!':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '!=')
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '!')

            if self.current_char == '<':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '<=')
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '<')

            if self.current_char == '>':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '>=')
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '>')

            if self.current_char == '{':
                self.advance()
                return Token(TokenType.LBRACE, '{')

            if self.current_char == '}':
                self.advance()
                return Token(TokenType.RBRACE, '}')

            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')')

            if self.current_char == ',':
                self.advance()
                return Token(TokenType.COMMA, ',')

            if self.current_char == ';':
                self.advance()
                return Token(TokenType.SEMICOLON, ';')
    
            if self.current_char in ['+', '-', '*', '/', '%']:
                op = self.current_char
                self.advance()
                return Token(TokenType.OPERATOR, op)

            self.error()

        return Token(TokenType.EOF, None)



    def boolean_expression(self):
        left = self.term()
        while self.current_token.type in (TokenType.OPERATOR):
            op = self.current_token
            self.eat(TokenType.OPERATOR)
            right = self.term()
            left = BinaryOperation(left, op, right)
        return left

    def variable(self):
        self.advance()  # Move past $
        token = self.identifier_token()  # Treat $ as part of identifier
        token.type = TokenType.VARIABLE  # Change token type to VARIABLE
        return token

    def peek(self):
        if self.pos + 1 < len(self.text):
            return self.text[self.pos + 1]
        return None

    def error(self):
        raise Exception(f"Invalid character: {self.current_char}")


class AST:
    pass

class Program(AST):
    def __init__(self):
        self.statements = []

class Print(AST):
    def __init__(self, expression):
        self.expr = expression

class Variable(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Input(AST):
    def __init__(self, prompt):
        self.prompt = prompt


class If(AST):
    def __init__(self, condition, body, else_body=None):
        self.condition = condition
        self.body = body
        self.else_body = else_body
         


class FunctionDef(AST):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class FunctionCall(AST):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        
class WhileStatement:
    def __init__(self, condition, body):
        self.condition = condition  # Condition is an expression
        self.body = body            # Body is a list of statements
       
class EchoStatement(AST):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"EchoStatement({self.expression})"

class Return(AST):
    def __init__(self, value):
        self.value = value
        
class BinaryOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class Literal(AST):
    def __init__(self, value):
        self.value = value

class Identifier(AST):
    def __init__(self, value):
        self.value = value

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type):
        #print(f"Eating token: {self.current_token}")  # Debug print
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f"Expected token type {token_type} but got {self.current_token.type}")
    def identifier(self):
        token = self.current_token
        self.eat(TokenType.IDENTIFIER)
        return Identifier(token.value)
    def statement(self):
        print(f"Parsing statement: {self.current_token}")
        if self.current_token.type == TokenType.PRINT:
            return self.print_statement()
        elif self.current_token.type == TokenType.VARIABLE:
            return self.variable_statement()
        elif self.current_token.type == TokenType.CUE:
            return self.cue_statement()
        elif self.current_token.type == TokenType.IF:
            return self.if_statement()
        elif self.current_token.type == TokenType.IDENTIFIER and self.peek() == TokenType.LPAREN:
            return self.function_call()
        elif self.current_token.type == TokenType.ECHO:
            return self.echo_statement()
        elif self.current_token.type == TokenType.WHILE:
            return self.while_statement
        else:
            self.error(f"Unexpected token: {self.current_token.type}")
    
    def error(self, message="Invalid syntax"):
        raise Exception(message)
        
    def while_statement(self):
        print(f"Parsing 'while' statement")
        self.eat(TokenType.WHILE)  # Consume 'while' token
        self.eat(TokenType.LPAREN)  # Consume '(' token
        
        condition = self.expression()  # Parse the loop condition
        print(f"Condition parsed: {condition}")
        self.eat(TokenType.RPAREN)  # Consume ')' token
        self.eat(TokenType.LBRACE)  # Consume '{' token
    
        body = []
        while self.current_token.type != TokenType.RBRACE:
            print(f"Parsing statement in while body")
            body.append(self.statement())  # Parse the statements within the loop
    
        self.eat(TokenType.RBRACE)  # Consume '}' token
    
        return WhileStatement(condition, body)

    def block(self):
        self.eat(TokenType.LBRACE)  # Consume '{' token
        statements = []
        while self.current_token.type != TokenType.RBRACE:
            statements.append(self.statement())  # Parse statements inside the block
        self.eat(TokenType.RBRACE)  # Consume '}' token
        return statements

    
    def function_def(self):
        self.eat(TokenType.FUNC)
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LPAREN)
    
        params = []
        if self.current_token.type == TokenType.IDENTIFIER:
            while self.current_token.type == TokenType.IDENTIFIER:
                params.append(self.current_token.value)
                self.eat(TokenType.IDENTIFIER)
                if self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)

        self.eat(TokenType.RPAREN)
        self.eat(TokenType.LBRACE)
    
        body = []
        while self.current_token.type != TokenType.RBRACE:
            body.append(self.statement())
            
        self.eat(TokenType.RBRACE)
    
        return FunctionDef(func_name, params, body)

    def function_body(self):
        statements = []
        while self.current_token.type != TokenType.RBRACE:
            if self.current_token.type == TokenType.RETURN:
                statements.append(self.return_statement())
            else:
                statements.append(self.statement())
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
        return statements

    def return_statement(self):
        self.eat(TokenType.RETURN)
        value = self.expression()
        return Return(value)

    def parameter_list(self):
        params = []
        if self.current_token.type == TokenType.IDENTIFIER:
            params.append(self.current_token.value)
            self.eat(TokenType.IDENTIFIER)
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                params.append(self.current_token.value)
                self.eat(TokenType.IDENTIFIER)
        return params

    def function_call(self):
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LPAREN)
    
        args = []
        if self.current_token.type != TokenType.RPAREN:
            while self.current_token.type != TokenType.RPAREN:
                args.append(self.expression())
                if self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)

        self.eat(TokenType.RPAREN)
    
        return FunctionCall(func_name, args)


    def argument_list(self):
        args = []
        if self.current_token.type in [TokenType.NUMBER, TokenType.STRING, TokenType.IDENTIFIER]:
            args.append(self.expression())
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.expression())
        return args
  

        
    def print_statement(self):
        if self.current_token.type == TokenType.ECHO:
            self.eat(TokenType.ECHO)
        else:
            self.eat(TokenType.PRINT)
        expr = self.expression()
        return Print(expr)
        
    def cue_statement(self):
        self.eat(TokenType.CUE)
        prompt = self.expression()
        return Input(prompt)
        
    def echo_statement(self):
        self.eat(TokenType.ECHO)
        expr = self.expression()  # Make sure you have a method to parse expressions
        # self.eat(TokenType.SEMICOLON)
        return EchoStatement(expr)
        
    def program(self):
        node = Program()
        while self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.RBRACE:
                break

            if self.current_token.type in [TokenType.PRINT, TokenType.ECHO]:
                node.statements.append(self.print_statement())
            elif self.current_token.type == TokenType.VARIABLE:
                node.statements.append(self.variable_statement())
            elif self.current_token.type == TokenType.CUE:
                node.statements.append(self.cue_statement())  
            elif self.current_token.type == TokenType.IF:
                node.statements.append(self.if_statement())
            elif self.current_token.type == TokenType.FUNC:
                node.statements.append(self.function_def())
            elif self.current_token.type == TokenType.IDENTIFIER and self.peek() == TokenType.LPAREN:
                node.statements.append(self.function_call())
            elif self.current_token.type == TokenType.WHILE:
                node.statements.append(self.while_statement())
            else:
                self.error(f"Unexpected token: {self.current_token.type}")

            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
            elif self.current_token.type == TokenType.EOF or self.current_token.type == TokenType.RBRACE:
                break
            else:
                self.error(f"Expected ';' or EOF but got {self.current_token.type}")

        return node

    def variable_statement(self):
        self.eat(TokenType.VARIABLE)
        name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ASSIGN)
        value = self.expression()
        return Variable(name, value)

    def input_statement(self):
        self.eat(TokenType.INPUT)
        prompt = self.expression()
        return Input(prompt)

        
    def if_statement(self):
        self.eat(TokenType.IF)
        condition = self.expression()
        self.eat(TokenType.LBRACE)
        body = self.program()
        self.eat(TokenType.RBRACE)

        else_body = None
        current_else = None

        while self.current_token.type in (TokenType.ELSEIF, TokenType.ELSE):
            if self.current_token.type == TokenType.ELSEIF:
                self.eat(TokenType.ELSEIF)
                elseif_condition = self.expression()
                self.eat(TokenType.LBRACE)
                elseif_body = self.program()
                self.eat(TokenType.RBRACE)
                new_if = If(elseif_condition, elseif_body, None)
                if else_body is None:
                    else_body = new_if
                    current_else = else_body
                else:
                    current_else.else_body = new_if
                    current_else = new_if
            elif self.current_token.type == TokenType.ELSE:
                self.eat(TokenType.ELSE)
                self.eat(TokenType.LBRACE)
                else_body_actual = self.program()
                self.eat(TokenType.RBRACE)
                if else_body is None:
                    else_body = else_body_actual
                else:
                    current_else.else_body = else_body_actual
                break

        return If(condition, body, else_body)

    def expression(self):
        return self.boolean_expression()

    def boolean_expression(self):
        expr = self.and_expression()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['==', '!=']:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.and_expression()
            expr = BinaryOp(expr, op, right)
        return expr

    def and_expression(self):
        expr = self.term()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value == '&&':
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.term()
            expr = BinaryOp(expr, op, right)
        return expr

    def term(self):
        expr = self.factor()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['+', '-']:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.factor()
            expr = BinaryOp(expr, op, right)
        return expr

    def factor(self):
        # Modify this method to call self.identifier() instead of creating Identifier node directly
        if self.current_token.type == TokenType.NUMBER:
            node = Literal(self.current_token.value)
            self.eat(TokenType.NUMBER)
            return node
        elif self.current_token.type == TokenType.STRING:
            node = Literal(self.current_token.value)
            self.eat(TokenType.STRING)
            return node
        elif self.current_token.type == TokenType.IDENTIFIER:
            return self.identifier()  # Updated to call self.identifier()
        elif self.current_token.type == TokenType.CUE:
            return self.cue_statement()  # Handle CUE statement here
        elif self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['!', '-']:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            expr = self.factor()
            return UnaryOp(op, expr)
        elif self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            expr = self.expression()
            self.eat(TokenType.RPAREN)
            return expr
        else:
            self.error(f"Unexpected factor: {self.current_token.type}")

    def parse(self):
        return self.program()

class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def set(self, name, value):
        self.symbols[name] = value

    def get(self, name):
        return self.symbols.get(name, None)

    def __str__(self):
        return f"SymbolTable({self.symbols})"


class Interpreter:
    def __init__(self, parser):
        self.parser = parser
        self.symbol_table = SymbolTable()

    def visit_program(self, program_node):
        for statement in program_node.statements:
            self.visit(statement)
    def visit_input(self, input_node):
        prompt_value = self.visit(input_node.prompt)
        user_input = input(prompt_value + " ")  # Prompt user for input
        return user_input

    def visit_print(self, print_node):
        value = self.visit(print_node.expr)
        print(value)
        

    def visit_variable(self, variable_node):
        value = self.visit(variable_node.value)
        self.symbol_table.set(variable_node.name, value)

    def visit_identifier(self, identifier_node):
            return self.symbol_table.get(identifier_node.value)

    def visit_literal(self, literal_node):
        return literal_node.value
        
    def visit_functiondef(self, functiondef_node):
        self.symbol_table.set(functiondef_node.name, functiondef_node)

    def visit_functioncall(self, functioncall_node):
        func = self.symbol_table.get(functioncall_node.name)
        if func is None:
            raise Exception(f"Function {functioncall_node.name} not defined")

        # Check parameter count and match arguments
        if len(func.params) != len(functioncall_node.args):
            raise Exception("Argument count mismatch")
        
        local_symbols = SymbolTable()
        for param, arg in zip(func.params, functioncall_node.args):
            local_symbols.set(param, self.visit(arg))
        
        # Execute function body
        old_symbol_table = self.symbol_table
        self.symbol_table = local_symbols
        result = None
        for statement in func.body:
            result = self.visit(statement)
            if isinstance(statement, Return):
                result = self.visit(statement.expr)
                break
        
        self.symbol_table = old_symbol_table
        return result
        
    def execute(self, statement):
        if isinstance(statement, WhileStatement):
            while self.evaluate(statement.condition):  # Evaluate the loop condition
                for stmt in statement.body:
                    self.execute(stmt)  # Execute each statement in the loop body
        # Handle other statement types

    def visit_return(self, return_node):
        return self.visit(return_node.value)
        
    def visit_binaryop(self, binaryop_node):
        left = self.visit(binaryop_node.left)
        right = self.visit(binaryop_node.right)
        if binaryop_node.op == '+':
            return left + right
        elif binaryop_node.op == '-':
            return left - right
        elif binaryop_node.op == '*':
            return left * right
        elif binaryop_node.op == '/':
            return left / right
        elif binaryop_node.op == '%':
            return left % right
        elif binaryop_node.op == '==':
            return left == right
        elif binaryop_node.op == '!=':
            return left != right
        else:
            raise Exception(f"Unknown operator {binaryop_node.op}")

    def visit_unaryop(self, unaryop_node):
        expr = self.visit(unaryop_node.expr)
        if unaryop_node.op == '-':
            return -expr
        elif unaryop_node.op == '!':
            return not expr
        else:
            raise Exception(f"Unknown unary operator {unaryop_node.op}")

    def visit_if(self, if_node):
        if self.visit(if_node.condition):
            self.visit_program(if_node.body)
        elif if_node.else_body:
            self.visit(if_node.else_body)


    def visit(self, node):
        method_name = f"visit_{type(node).__name__.lower()}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__.lower()} method defined")
        


def run(source):
    lexer = Lexer(source)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    program = parser.program()
    interpreter.visit_program(program)

def main():
    if len(sys.argv) != 2:
        print("Usage: tuelang <filename.tuelang>")
        return

    filename = sys.argv[1]

    try:
        with open(filename, 'r') as file:
            source_code = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    run(source_code)

if __name__ == "__main__":
    main()
