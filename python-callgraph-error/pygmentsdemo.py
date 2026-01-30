from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from pygments.lexers.matlab import MatlabLexer
from pygments.lexers import (get_lexer_by_name,
    get_lexer_for_filename, get_lexer_for_mimetype)
import re

def lexerDemo():
    code = """
    def greet(name):
        print("Hello, " + name + "!")

    greet("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    """

    lexer = PythonLexer()
    formatter = TerminalFormatter()

    highlighted_code = highlight(code, lexer, formatter)
    print(highlighted_code)

def  matlabDemo():
    lexer = MatlabLexer()
    lexer.analyse_text("test")  
    code = """
    % MATLAB example code
    x = linspace(0, 2*pi, 100);
    y = sin(x);
    plot(x, y);
    """

    lexer = MatlabLexer()
    formatter = TerminalFormatter()

    highlighted_code = highlight(code, lexer, formatter)
    print(highlighted_code) 

# def validate_input(input_string):
#     pattern = r"^(a+)+$"
#     match = re.match(pattern, input_string)
    
#     if match:
#         print("Valid input")
#     else:
#         print("Invalid input")


if __name__ == '__main__':
    lexerDemo()

    # Malicious input triggering ReDoS
    input_string = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!"

    # validate_input(input_string) 
    get_lexer_by_name('python')  
    matlabDemo() 

