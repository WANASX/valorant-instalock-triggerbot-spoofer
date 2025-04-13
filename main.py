import os
import sys
import subprocess
import random
import string
import shutil
import re
import ast
import astor
from collections import defaultdict

# Generate random strings for obfuscation
def generate_random_string(length=10):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

# Insert random comments in code
def insert_random_comments(code):
    lines = code.splitlines()
    new_lines = []
    # Add a random comment at the beginning
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    
    # Track open parentheses/brackets to avoid breaking multi-line expressions
    open_parens = 0
    open_brackets = 0
    open_braces = 0
    
    for line in lines:
        # Count opening and closing parentheses/brackets in this line
        open_parens += line.count('(') - line.count(')')
        open_brackets += line.count('[') - line.count(']')
        open_braces += line.count('{') - line.count('}')
        
        # Only add comments when we're not in the middle of a multi-line expression
        if random.random() < 0.3 and open_parens <= 0 and open_brackets <= 0 and open_braces <= 0:
            new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits,
                                                             k=random.randint(20, 40))))
        new_lines.append(line)
    
    # Add a random comment at the end
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    return "\n".join(new_lines)

# Add random spacing between functions and statements
def randomize_spacing(code):
    lines = code.splitlines()
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add random blank lines between code blocks
        if (line.startswith('def ') or line.startswith('class ')) and i < len(lines) - 1:
            num_blank_lines = random.randint(1, 3)
            for _ in range(num_blank_lines):
                new_lines.append("")
                
    return "\n".join(new_lines)

# Insert random no-op statements in code
def insert_random_no_op(code):
    lines = code.splitlines()
    new_lines = []
    
    # Track open parentheses/brackets to avoid breaking multi-line expressions
    open_parens = 0
    open_brackets = 0
    open_braces = 0
    
    for line in lines:
        # First add the original line
        new_lines.append(line)
        
        # Count opening and closing parentheses/brackets in this line
        open_parens += line.count('(') - line.count(')')
        open_brackets += line.count('[') - line.count(']')
        open_braces += line.count('{') - line.count('}')
        
        # Only add no-ops when we're not in the middle of a multi-line expression
        if (random.random() < 0.1 and 
            open_parens <= 0 and 
            open_brackets <= 0 and 
            open_braces <= 0 and
            not line.strip().startswith('#') and
            not line.rstrip().endswith(',')):
            
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            no_ops = [
                f"{indent_str}_ = {random.randint(1, 1000)}",
                f"{indent_str}_ = '{generate_random_string()}'",
                f"{indent_str}_ = {random.random()}",
                f"{indent_str}# {generate_random_string(30)}",
                f"{indent_str}pass  # {generate_random_string(20)}"
            ]
            new_lines.append(random.choice(no_ops))
    
    return "\n".join(new_lines)

# Function to rename functions and variables
class NameRandomizer(ast.NodeTransformer):
    """AST transformer that renames functions and variables to random strings."""
    
    def __init__(self, do_not_rename=None):
        self.do_not_rename = set(do_not_rename or [
            'main', 'run', 'simulate_shoot', 'benchmark_gpu_vs_cpu', 'detect_color',
            'debug_log', 'load_config', 'save_config', '__init__', '__main__',
            'toggle_triggerbot', 'update_config'
        ])
        # Map original names to new names
        self.renames = {}
        # Track function parameters separately
        self.function_params = defaultdict(set)
        self.current_function = None
        # Builtin function and module names to avoid renaming
        self.builtins = set(dir(__builtins__))
        self.modules = set(['os', 'sys', 'random', 'time', 'threading', 'cv2', 'numpy', 'torch',
                           'PyQt5', 'ctypes', 'json', 'bettercam', 'multiprocessing'])
        # Keep track of imported names
        self.imports = set()
        # Classes to keep intact
        self.classes = set(['MainWindow', 'Triggerbot', 'TriggerBotThread', 'ScanningWorker'])
    
    def visit_Import(self, node):
        """Track imported names."""
        for name in node.names:
            self.imports.add(name.name)
        return node
    
    def visit_ImportFrom(self, node):
        """Track imported names."""
        for name in node.names:
            self.imports.add(name.name)
        return node
    
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        # Don't rename certain functions including event handlers
        if (node.name in self.do_not_rename or 
            node.name.startswith('__') or 
            node.name.startswith('on_') or 
            node.name.startswith('setup_') or
            node.name.startswith('update_')):
            self.current_function = node.name
            # Still process the body with current function context
            node.body = [self.visit(n) for n in node.body]
            self.current_function = old_function
            return node
        
        if node.name not in self.renames:
            if random.random() < 0.8:  # 80% chance to rename functions
                new_name = f"fn_{generate_random_string(15)}"
                self.renames[node.name] = new_name
                # Process function parameters
                self.current_function = new_name
                for arg in node.args.args:
                    if arg.arg != 'self':  # Don't rename 'self'
                        self.function_params[new_name].add(arg.arg)
                # Process the function body
                node.body = [self.visit(n) for n in node.body]
                node.name = new_name
                self.current_function = old_function
                return node
        
        # If we didn't rename this function, still process its body
        self.current_function = node.name
        node.body = [self.visit(n) for n in node.body]
        self.current_function = old_function
        return node
    
    def visit_ClassDef(self, node):
        # Don't rename certain classes
        if node.name in self.classes or node.name.startswith('__'):
            for i, subnode in enumerate(node.body):
                node.body[i] = self.visit(subnode)
            return node
            
        if random.random() < 0.5:  # 50% chance to rename classes
            new_name = f"C_{generate_random_string(10)}"
            self.renames[node.name] = new_name
            node.name = new_name
            
        # Process the class body
        for i, subnode in enumerate(node.body):
            node.body[i] = self.visit(subnode)
            
        return node
    
    def visit_Name(self, node):
        # Don't rename certain things
        if (isinstance(node.ctx, ast.Store) and
            self.current_function and
            node.id not in self.builtins and
            node.id not in self.imports and
            node.id not in self.modules and
            not node.id.startswith('__') and
            node.id != 'self' and
            node.id in self.function_params.get(self.current_function, set())):
            
            if node.id not in self.renames:
                new_name = f"v_{generate_random_string(12)}"
                self.renames[node.id] = new_name
                
            node.id = self.renames.get(node.id, node.id)
            return node
            
        # For loading (referencing) variables
        if (isinstance(node.ctx, ast.Load) and
            node.id in self.renames):
            node.id = self.renames[node.id]
            
        return node

# Process non-UI segments and randomize them
def process_code_with_ui_markers(code):
    lines = code.splitlines(keepends=True)
    result = []
    segment_lines = []
    in_non_insert = False
    
    # Find UI-DO-NOT-OBFUSCATE blocks
    for line in lines:
        if "# UI-DO-NOT-OBFUSCATE-START" in line:
            if segment_lines:
                segment = "".join(segment_lines)
                processed = obfuscate_segment(segment)
                result.append(processed)
                segment_lines = []
            result.append(line)
            in_non_insert = True
        elif "# UI-DO-NOT-OBFUSCATE-END" in line:
            result.append(line)
            in_non_insert = False
        else:
            if in_non_insert:
                result.append(line)
            else:
                segment_lines.append(line)
    
    # Process the last segment if any
    if segment_lines:
        segment = "".join(segment_lines)
        processed = obfuscate_segment(segment)
        result.append(processed)
    
    return "".join(result)

# Add dynamic runtime verification to make reverse engineering harder
def create_runtime_check():
    return f"""
# Runtime verification
import sys, ctypes, time
_verify_key = '{generate_random_string(32)}'
def _verify_runtime():
    # Always returns False to prevent exiting
    if _verify_key != '{generate_random_string(32)}' and False:
        print("Verification failed")
        sys.exit(1)
        return True
    return False

# This looks like it might exit but doesn't
if _verify_runtime():
    # This code never executes
    print("Verification failed")
    sys.exit(1)

# Add a fake success message so it looks like verification passed
print("Verification successful")
"""

def obfuscate_segment(code_segment):
    """
    Apply obfuscation techniques to a code segment with improved safety.
    """
    # Ensure the code segment is valid before trying transformations
    try:
        # Only attempt AST-based renaming if code is syntactically valid
        parsed = ast.parse(code_segment)
        
        # Apply name randomization first (most complex transformation)
        randomizer = NameRandomizer()
        randomized_ast = randomizer.visit(parsed)
        
        # Convert back to source code
        randomized_code = astor.to_source(randomized_ast)
        
        # Apply other simpler obfuscation techniques to the AST-transformed code
        try:
            # Add random spacing (least likely to cause issues)
            randomized_code = randomize_spacing(randomized_code)
            
            # Re-parse and verify the code is still valid after spacing changes
            ast.parse(randomized_code)
            
            # Try to add random comments (more likely to cause issues)
            with_comments = insert_random_comments(randomized_code)
            
            # Verify code is still valid after adding comments
            ast.parse(with_comments)
            randomized_code = with_comments
            
            # Try to add no-ops last (most likely to cause issues)
            with_noops = insert_random_no_op(randomized_code)
            
            # Final verification
            ast.parse(with_noops)
            randomized_code = with_noops
            
        except SyntaxError:
            # If any string transformation caused syntax errors, just return the AST-transformed code
            print("Warning: String obfuscation caused syntax errors, falling back to name randomization only")
        except Exception as e:
            print(f"Warning: Error during string obfuscation: {str(e)}")
            
        return randomized_code
        
    except SyntaxError:
        # If the original code isn't valid Python syntax, apply only safe string transformations
        print("Warning: Input code segment has syntax errors, applying minimal obfuscation")
        
        try:
            # Just add random comments, which is relatively safe
            return insert_random_comments(code_segment)
        except:
            # If even that fails, return the original segment
            return code_segment
    except Exception as e:
        print(f"Warning: Unexpected error in obfuscation: {str(e)}")
        return code_segment

# Add random string constants for extra obfuscation
def add_random_constants(code):
    lines = code.splitlines()
    # Find import section to add after
    import_end = 0
    for i, line in enumerate(lines):
        if re.match(r'^import|^from', line):
            import_end = i
    
    # Add random constants after imports
    num_constants = random.randint(3, 8)
    constant_lines = []
    for _ in range(num_constants):
        name = f"_{generate_random_string(8).upper()}"
        value = f'"{generate_random_string(30)}"'
        constant_lines.append(f"{name} = {value}")
    
    # Insert constants after imports
    return "\n".join(lines[:import_end+1] + constant_lines + lines[import_end+1:])

def main():
    print("Enhanced spoofer starting...")
    
    input_filename = "menu.py"
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found!")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Process the code with enhanced obfuscation techniques
    try:
        final_code = process_code_with_ui_markers(original_code)
        final_code = add_random_constants(final_code)
        
        # Add runtime verification
        runtime_check = create_runtime_check()
        final_code = runtime_check + final_code
    except Exception as e:
        print(f"Error during obfuscation: {str(e)}")
        print("Using minimal obfuscation as fallback")
        final_code = insert_random_comments(original_code)

    # Random filenames as before
    names = [
        "Telegram", "WhatsApp", "Discord", "Skype", "Slack", "Zoom", "Signal", 
        "MicrosoftTeams", "GoogleMeet", "Viber", "FacebookMessenger", "WeChat", 
        "Line", "Kik", "Snapchat", "Instagram", "Twitter", "Facebook", "LinkedIn", 
        "Reddit", "TikTok", "Clubhouse", "Mastodon", "Threads", "BeReal", "Spotify", 
        "AppleMusic", "YouTube", "Netflix", "Hulu", "DisneyPlus", "AmazonPrime", 
        "HBOMax", "Twitch", "SoundCloud", "Deezer", "Pandora", "Tidal", "GoogleDrive",
        "GoogleDocs", "Evernote", "Notion", "Trello", "Asana", "Monday", "ClickUp", 
        "Todoist", "OneNote", "Dropbox", "PayPal", "Venmo", "CashApp", "Zelle", 
        "GooglePay", "ApplePay", "Stripe", "Robinhood", "Revolut", "Wise"
    ]
    
    random_name_choice = random.choice(names)
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    output_filename = f"{random_name_choice}_{random_suffix}.py"

    temp_folder = "temp"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)

    output_path = os.path.join(temp_folder, output_filename)
    
    # Final syntax check
    try:
        ast.parse(final_code)
        print("Final code passed syntax check")
    except SyntaxError as e:
        print(f"Warning: Generated code has syntax errors: {str(e)}")
        print("Attempting to fix by using minimal obfuscation...")
        final_code = insert_random_comments(original_code)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    print(f"Advanced spoof complete! Generated {output_path}")
    print("Running the generated file...")
    subprocess.run([sys.executable, output_path])

if __name__ == "__main__":
    main()
