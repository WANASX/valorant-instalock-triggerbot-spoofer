import re
import random
import string
import os
import subprocess
import sys

def generate_random_name(prefix="f_"):
    """Generate a random name with the given prefix."""
    return prefix + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def random_comment_line():
    """Generate a random comment line to inject as dead code."""
    length = random.randint(20, 40)
    comment = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return "# " + comment

def randomize_import_aliases(code):
    """
    Randomize the import aliases in the code for simple import statements.
    Only lines of the form 'import module' (or with an alias) are processed.
    In a second pass, only non-import lines have their usages replaced.
    """
    lines = code.splitlines()
    mapping = {}  # mapping from original alias to new alias
    new_lines = []
    
    # First pass: update simple import lines and build the mapping.
    for line in lines:
        m = re.match(r'^(import\s+)([a-zA-Z0-9_.]+)(\s+as\s+([a-zA-Z0-9_]+))?\s*$', line)
        if m:
            prefix = m.group(1)  # "import "
            module_name = m.group(2)  # e.g., "cv2"
            # Determine the original alias: if an alias is provided, use it; otherwise, use the module's base name.
            orig_alias = m.group(4) if m.group(4) else module_name.split('.')[-1]
            new_alias = generate_random_name(orig_alias + "_")
            mapping[orig_alias] = new_alias
            new_line = f"{prefix}{module_name} as {new_alias}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    # Second pass: For lines that are not import lines, replace usages of the original aliases.
    final_lines = []
    for line in new_lines:
        if re.match(r'^\s*(import|from)\s+', line):
            # Do not modify lines that start with 'import' or 'from'
            final_lines.append(line)
        else:
            for old_alias, new_alias in mapping.items():
                # Replace only tokens not preceded by a dot to avoid altering attribute names.
                line = re.sub(r'(?<!\.)\b' + re.escape(old_alias) + r'\b', new_alias, line)
            final_lines.append(line)
    
    return "\n".join(final_lines)

def obfuscate_code(code):
    """
    Obfuscate the given source code by:
      0. Randomizing import aliases.
      1. Renaming top-level functions and classes.
      2. Inserting random comment lines at random locations.
    """
    # Step 0: Randomize import aliases.
    code = randomize_import_aliases(code)
    
    # --- Step 1: Rename functions and classes ---
    function_names = re.findall(r'def\s+(\w+)\(', code)
    class_names = re.findall(r'class\s+(\w+)\(', code)
    rename_map = {}
    
    # Rename functions (skip dunder names)
    for name in set(function_names):
        if name.startswith("__"):
            continue
        new_name = generate_random_name("fn_")
        rename_map[name] = new_name

    # Rename classes
    for name in set(class_names):
        if name.startswith("__"):
            continue
        new_name = generate_random_name("cl_")
        rename_map[name] = new_name

    # Replace all occurrences with new names.
    for old_name, new_name in rename_map.items():
        pattern = r'\b' + re.escape(old_name) + r'\b'
        code = re.sub(pattern, new_name, code)
    
    # --- Step 2: Insert random comment lines ---
    lines = code.splitlines()
    obfuscated_lines = []
    for line in lines:
        # With a 30% chance, insert a random comment line before the current line.
        if random.random() < 0.3:
            obfuscated_lines.append(random_comment_line())
        obfuscated_lines.append(line)
    # Insert a random comment at the beginning and the end.
    obfuscated_lines.insert(0, random_comment_line())
    obfuscated_lines.append(random_comment_line())
    
    return "\n".join(obfuscated_lines)

def main():
    input_filename = "Telegram.py"

    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found!")
        return

    # Read the original source code.
    with open(input_filename, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Obfuscate the code.
    spoofed_code = obfuscate_code(original_code)

    # Generate a random output filename, e.g., Telegram_ab12cd34.py.
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    output_filename = f"Telegram_{random_suffix}.py"

    # Write the obfuscated code to the new file.
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(spoofed_code)

    print(f"Spoof complete! Generated {output_filename}")

    # Run the newly generated file using the current Python interpreter.
    print("Running the generated file...")
    subprocess.run([sys.executable, output_filename])

if __name__ == "__main__":
    main()
