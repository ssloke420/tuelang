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
    FOR = 'FOR'
    BREAK = 'BREAK'
    CONTINUE = 'CONTINUE'

class Token:
    def __init__(self, type, value, line=0, column=0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, {self.line}:{self.column})"

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def advance(self):
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
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
        line, column = self.line, self.column
        self.advance()  # Skip the opening quote
        while self.current_char and self.current_char != '"':
            if self.current_char == '\\':  # Handle escape sequences
                self.advance()
                if self.current_char == 'n':
                    result += '\n'
                elif self.current_char == 't':
                    result += '\t'
                elif self.current_char == 'r':
                    result += '\r'
                elif self.current_char == '\\':
                    result += '\\'
                elif self.current_char == '"':
                    result += '"'
                else:
                    result += self.current_char
            else:
                result += self.current_char
            self.advance()
        
        if self.current_char != '"':
            raise Exception(f"Unterminated string at line {line}, column {column}")
        
        self.advance()  # Skip the closing quote
        return Token(TokenType.STRING, result, line, column)

    def identifier_token(self):
        result = ''
        line, column = self.line, self.column
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        # Map the identifier to the appropriate token type
        keywords = {
            'cue': TokenType.CUE,
            'echo': TokenType.ECHO,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'elseif': TokenType.ELSEIF,
            'func': TokenType.FUNC,
            'return': TokenType.RETURN,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
        }
        
        token_type = keywords.get(result.lower(), TokenType.IDENTIFIER)
        return Token(token_type, result, line, column)

    def number(self):
        result = ''
        line, column = self.line, self.column
        while self.current_char and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        if self.current_char == '.':
            result += '.'
            self.advance()
            while self.current_char and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(TokenType.NUMBER, float(result), line, column)
        return Token(TokenType.NUMBER, int(result), line, column)

    def get_next_token(self):
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # Handle ~~ comments
            if self.current_char == '~' and self.peek() == '~':
                self.advance()
                self.advance()
                if self.current_char == '\n':
                    self.skip_comment()
                else:
                    while self.current_char and (self.current_char != '~' or self.peek() != '~'):
                        self.advance()
                    if self.current_char:
                        self.advance()
                    if self.current_char:
                        self.advance()
                continue

            line, column = self.line, self.column

            if self.current_char == '"':
                return self.string()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier_token()

            if self.current_char == '$':
                return self.variable()

            # Enhanced operator handling
            if self.current_char == '=':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '==', line, column)
                else:
                    self.advance()
                    return Token(TokenType.ASSIGN, '=', line, column)

            if self.current_char == '!':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '!=', line, column)
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '!', line, column)

            if self.current_char == '<':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '<=', line, column)
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '<', line, column)

            if self.current_char == '>':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '>=', line, column)
                else:
                    self.advance()
                    return Token(TokenType.OPERATOR, '>', line, column)

            # Handle && and ||
            if self.current_char == '&':
                if self.peek() == '&':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '&&', line, column)

            if self.current_char == '|':
                if self.peek() == '|':
                    self.advance()
                    self.advance()
                    return Token(TokenType.OPERATOR, '||', line, column)

            # Single character tokens
            single_char_tokens = {
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ',': TokenType.COMMA,
                ';': TokenType.SEMICOLON,
                '+': TokenType.OPERATOR,
                '-': TokenType.OPERATOR,
                '*': TokenType.OPERATOR,
                '/': TokenType.OPERATOR,
                '%': TokenType.OPERATOR,
            }

            if self.current_char in single_char_tokens:
                char = self.current_char
                token_type = single_char_tokens[char]
                self.advance()
                if token_type == TokenType.OPERATOR:
                    return Token(token_type, char, line, column)
                else:
                    return Token(token_type, char, line, column)

            self.error()

        return Token(TokenType.EOF, None, self.line, self.column)

    def variable(self):
        line, column = self.line, self.column
        self.advance()  # Move past $
        token = self.identifier_token()  # Treat $ as part of identifier
        token.type = TokenType.VARIABLE  # Change token type to VARIABLE
        token.line = line
        token.column = column
        return token

    def peek(self):
        if self.pos + 1 < len(self.text):
            return self.text[self.pos + 1]
        return None

    def error(self):
        raise Exception(f"Invalid character: '{self.current_char}' at line {self.line}, column {self.column}")


# AST Node classes
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
        
class WhileStatement(AST):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStatement(AST):
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

class BreakStatement(AST):
    pass

class ContinueStatement(AST):
    pass

class EchoStatement(AST):
    def __init__(self, expression):
        self.expression = expression

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

# Exception classes for control flow
class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f"Expected token type {token_type} but got {self.current_token.type} at line {self.current_token.line}")

    def peek(self):
        # Save current state
        saved_pos = self.lexer.pos
        saved_char = self.lexer.current_char
        saved_line = self.lexer.line
        saved_column = self.lexer.column
        
        # Get next token
        next_token = self.lexer.get_next_token()
        
        # Restore state
        self.lexer.pos = saved_pos
        self.lexer.current_char = saved_char
        self.lexer.line = saved_line
        self.lexer.column = saved_column
        
        return next_token.type

    def error(self, message="Invalid syntax"):
        raise Exception(f"{message} at line {self.current_token.line}")

    def identifier(self):
        token = self.current_token
        self.eat(TokenType.IDENTIFIER)
        return Identifier(token.value)

    def statement(self):
        if self.current_token.type in [TokenType.PRINT, TokenType.ECHO]:
            return self.print_statement()
        elif self.current_token.type == TokenType.VARIABLE:
            return self.variable_statement()
        elif self.current_token.type == TokenType.CUE:
            return self.cue_statement()
        elif self.current_token.type == TokenType.IF:
            return self.if_statement()
        elif self.current_token.type == TokenType.FUNC:
            return self.function_def()
        elif self.current_token.type == TokenType.IDENTIFIER and self.peek() == TokenType.LPAREN:
            return self.function_call()
        elif self.current_token.type == TokenType.WHILE:
            return self.while_statement()
        elif self.current_token.type == TokenType.FOR:
            return self.for_statement()
        elif self.current_token.type == TokenType.RETURN:
            return self.return_statement()
        elif self.current_token.type == TokenType.BREAK:
            self.eat(TokenType.BREAK)
            return BreakStatement()
        elif self.current_token.type == TokenType.CONTINUE:
            self.eat(TokenType.CONTINUE)
            return ContinueStatement()
        else:
            self.error(f"Unexpected token: {self.current_token.type}")

    def while_statement(self):
        self.eat(TokenType.WHILE)
        self.eat(TokenType.LPAREN)
        condition = self.expression()
        self.eat(TokenType.RPAREN)
        self.eat(TokenType.LBRACE)
        
        body = []
        while self.current_token.type != TokenType.RBRACE:
            body.append(self.statement())
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
        
        self.eat(TokenType.RBRACE)
        return WhileStatement(condition, body)

    def for_statement(self):
        self.eat(TokenType.FOR)
        self.eat(TokenType.LPAREN)
        
        # Initialization
        init = None
        if self.current_token.type != TokenType.SEMICOLON:
            init = self.statement()
        self.eat(TokenType.SEMICOLON)
        
        # Condition
        condition = None
        if self.current_token.type != TokenType.SEMICOLON:
            condition = self.expression()
        self.eat(TokenType.SEMICOLON)
        
        # Update
        update = None
        if self.current_token.type != TokenType.RPAREN:
            update = self.expression()  # Simple expression for update
        self.eat(TokenType.RPAREN)
        
        # Body
        self.eat(TokenType.LBRACE)
        body = []
        while self.current_token.type != TokenType.RBRACE:
            body.append(self.statement())
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
        self.eat(TokenType.RBRACE)
        
        return ForStatement(init, condition, update, body)

    def function_def(self):
        self.eat(TokenType.FUNC)
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LPAREN)
    
        params = []
        if self.current_token.type == TokenType.IDENTIFIER:
            params.append(self.current_token.value)
            self.eat(TokenType.IDENTIFIER)
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                params.append(self.current_token.value)
                self.eat(TokenType.IDENTIFIER)

        self.eat(TokenType.RPAREN)
        self.eat(TokenType.LBRACE)
    
        body = []
        while self.current_token.type != TokenType.RBRACE:
            body.append(self.statement())
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
            
        self.eat(TokenType.RBRACE)
        return FunctionDef(func_name, params, body)

    def return_statement(self):
        self.eat(TokenType.RETURN)
        value = None
        if self.current_token.type not in [TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF]:
            value = self.expression()
        return Return(value)

    def function_call(self):
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LPAREN)
    
        args = []
        if self.current_token.type != TokenType.RPAREN:
            args.append(self.expression())
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.expression())

        self.eat(TokenType.RPAREN)
        return FunctionCall(func_name, args)

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

    def program(self):
        node = Program()
        while self.current_token.type != TokenType.EOF and self.current_token.type != TokenType.RBRACE:
            node.statements.append(self.statement())
            
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
            elif self.current_token.type in [TokenType.EOF, TokenType.RBRACE]:
                break

        return node

    def variable_statement(self):
        self.eat(TokenType.VARIABLE)
        name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ASSIGN)
        value = self.expression()
        return Variable(name, value)

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
        return self.or_expression()

    def or_expression(self):
        expr = self.and_expression()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value == '||':
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.and_expression()
            expr = BinaryOp(expr, op, right)
        return expr

    def and_expression(self):
        expr = self.equality_expression()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value == '&&':
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.equality_expression()
            expr = BinaryOp(expr, op, right)
        return expr

    def equality_expression(self):
        expr = self.comparison_expression()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['==', '!=']:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.comparison_expression()
            expr = BinaryOp(expr, op, right)
        return expr

    def comparison_expression(self):
        expr = self.term()
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in ['<', '>', '<=', '>=']:
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
        if self.current_token.type == TokenType.NUMBER:
            node = Literal(self.current_token.value)
            self.eat(TokenType.NUMBER)
            return node
        elif self.current_token.type == TokenType.STRING:
            node = Literal(self.current_token.value)
            self.eat(TokenType.STRING)
            return node
        elif self.current_token.type == TokenType.IDENTIFIER:
            if self.peek() == TokenType.LPAREN:
                return self.function_call()
            else:
                return self.identifier()
        elif self.current_token.type == TokenType.CUE:
            return self.cue_statement()
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
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def set(self, name, value):
        self.symbols[name] = value

    def get(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.get(name)
        return None

    def __str__(self):
        return f"SymbolTable({self.symbols})"

class Interpreter:
    def __init__(self, parser):
        self.parser = parser
        self.symbol_table = SymbolTable()

    def visit_program(self, program_node):
        result = None
        for statement in program_node.statements:
            result = self.visit(statement)
        return result

    def visit_input(self, input_node):
        prompt_value = self.visit(input_node.prompt)
        user_input = input(str(prompt_value) + " ")
        # Try to convert to number if possible
        try:
            if '.' in user_input:
                return float(user_input)
            else:
                return int(user_input)
        except ValueError:
            return user_input

    def visit_print(self, print_node):
        value = self.visit(print_node.expr)
        print(value)

    def visit_variable(self, variable_node):
        value = self.visit(variable_node.value)
        self.symbol_table.set(variable_node.name, value)

    def visit_identifier(self, identifier_node):
        value = self.symbol_table.get(identifier_node.value)
        if value is None:
            raise Exception(f"Undefined variable: {identifier_node.value}")
        return value

    def visit_literal(self, literal_node):
        return literal_node.value
        
    def visit_functiondef(self, functiondef_node):
        self.symbol_table.set(functiondef_node.name, functiondef_node)

    def visit_functioncall(self, functioncall_node):
        func = self.symbol_table.get(functioncall_node.name)
        if func is None:
            raise Exception(f"Function {functioncall_node.name} not defined")

        if not isinstance(func, FunctionDef):
            raise Exception(f"{functioncall_node.name} is not a function")

        # Check parameter count
        if len(func.params) != len(functioncall_node.args):
            raise Exception(f"Function {functioncall_node.name} expects {len(func.params)} arguments, got {len(functioncall_node.args)}")
        
        # Create new scope
        local_symbols = SymbolTable(self.symbol_table)
        for param, arg in zip(func.params, functioncall_node.args):
            local_symbols.set(param, self.visit(arg))
        
        # Execute function body
        old_symbol_table = self.symbol_table
        self.symbol_table = local_symbols
        result = None
        
        try:
            for statement in func.body:
                result = self.visit(statement)
        except ReturnException as ret:
            result = ret.value
        finally:
            self.symbol_table = old_symbol_table
        
        return result

    def visit_whilestatement(self, while_node):
        try:
            while self.visit(while_node.condition):
                try:
                    for statement in while_node.body:
                        self.visit(statement)
                except ContinueException:
                    continue
                except BreakException:
                    break
        except BreakException:
            pass

    def visit_forstatement(self, for_node):
        # Execute initialization
        if for_node.init:
            self.visit(for_node.init)
        
        try:
            while True:
                # Check condition
                if for_node.condition and not self.visit(for_node.condition):
                    break
                
                try:
                    # Execute body
                    for statement in for_node.body:
                        self.visit(statement)
                except ContinueException:
                    pass
                except BreakException:
                    break
                
                # Execute update
                if for_node.update:
                    self.visit(for_node.update)
        except BreakException:
            pass

    def visit_breakstatement(self, break_node):
        raise BreakException()

    def visit_continuestatement(self, continue_node):
        raise ContinueException()

    def visit_return(self, return_node):
        value = None
        if return_node.value:
            value = self.visit(return_node.value)
        raise ReturnException(value)
        
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
            if right == 0:
                raise Exception("Division by zero")
            return left / right
        elif binaryop_node.op == '%':
            return left % right
        elif binaryop_node.op == '==':
            return left == right
        elif binaryop_node.op == '!=':
            return left != right
        elif binaryop_node.op == '<':
            return left < right
        elif binaryop_node.op == '>':
            return left > right
        elif binaryop_node.op == '<=':
            return left <= right
        elif binaryop_node.op == '>=':
            return left >= right
        elif binaryop_node.op == '&&':
            return left and right
        elif binaryop_node.op == '||':
            return left or right
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
            if isinstance(if_node.else_body, If):
                self.visit(if_node.else_body)
            else:
                self.visit_program(if_node.else_body)

    def visit_echostatement(self, echo_node):
        value = self.visit(echo_node.expression)
        print(value)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__.lower()}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__.lower()} method defined for {type(node).__name__}")

def run(source):
    try:
        lexer = Lexer(source)
        parser = Parser(lexer)
        program = parser.parse()
        interpreter = Interpreter(parser)
        interpreter.visit_program(program)
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python tuelang.py <filename.tuelang>")
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
