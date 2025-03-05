import os
import sys
import subprocess
import random
import string
import shutil
import tokenize
import io
import keyword

def generate_random_name(prefix="f_"):
    return prefix + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def obfuscate_code_tokens(code):
    """
    Renames all identifier tokens (type NAME) that are not in the reserved list or
    immediately following a dot (attribute access). This version now processes all tokens,
    including those on import lines, but preserves module names and other reserved identifiers.
    """
    reserved = set(keyword.kwlist)
    try:
        reserved |= set(__builtins__.keys())
    except Exception:
        reserved |= set(dir(__builtins__))
    reserved |= {
        "self", "sys", "os", "time", "random", "ctypes", "json", "threading",
        "Process", "Queue", "cv2", "numpy", "np", "pyautogui", "win32api", "win32con",
        "QtWidgets", "QtCore", "QtGui", "bettercam", "shutdown_event", "subprocess", "shutil",
        "PyQt5", "multiprocessing"
    }
    mapping = {}
    tokens = []
    sio = io.StringIO(code)
    prev_token = None
    for tok in tokenize.generate_tokens(sio.readline):
        if tok.type == tokenize.NAME:
            # Avoid obfuscating attributes (tokens immediately following a dot)
            if prev_token is not None and prev_token.type == tokenize.OP and prev_token.string == '.':
                tokens.append(tok)
            elif tok.string not in reserved:
                if tok.string not in mapping:
                    mapping[tok.string] = generate_random_name("_")
                new_tok = tokenize.TokenInfo(tok.type, mapping[tok.string], tok.start, tok.end, tok.line)
                tokens.append(new_tok)
            else:
                tokens.append(tok)
        else:
            tokens.append(tok)
        prev_token = tok
    return tokenize.untokenize(tokens)

def insert_random_comments(code):
    """
    Inserts random comment lines at random positions in the code.
    This makes the obfuscated output less predictable.
    """
    lines = code.splitlines()
    new_lines = []
    # Optionally add a random comment at the beginning
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    for line in lines:
        if random.random() < 0.3:
            new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits,
                                                             k=random.randint(20, 40))))
        new_lines.append(line)
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    return "\n".join(new_lines)

def obfuscate_segment(code):
    """
    Applies token obfuscation and then inserts random comments.
    Assumes the entire segment should be obfuscated.
    """
    code = obfuscate_code_tokens(code)
    code = insert_random_comments(code)
    return code

def obfuscate_code_with_ui_markers(code):
    """
    Processes the entire file by splitting it into segments.
    Any code between the markers:
      # UI-DO-NOT-OBFUSCATE-START
      and
      # UI-DO-NOT-OBFUSCATE-END
    is left intact, while all other segments are obfuscated.
    """
    lines = code.splitlines(keepends=True)
    result = []
    obf_segment_lines = []
    in_non_obf = False
    for line in lines:
        if "# UI-DO-NOT-OBFUSCATE-START" in line:
            if obf_segment_lines:
                segment = "".join(obf_segment_lines)
                result.append(obfuscate_segment(segment))
                obf_segment_lines = []
            result.append(line)
            in_non_obf = True
        elif "# UI-DO-NOT-OBFUSCATE-END" in line:
            result.append(line)
            in_non_obf = False
        else:
            if in_non_obf:
                result.append(line)
            else:
                obf_segment_lines.append(line)
    if obf_segment_lines:
        segment = "".join(obf_segment_lines)
        result.append(obfuscate_segment(segment))
    return "".join(result)

def main():
    input_filename = "menu.py"
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found!")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        original_code = f.read()

    final_code = obfuscate_code_with_ui_markers(original_code)

    names = [
        "Telegram", "WhatsApp", "Discord", "Skype", "Slack", "Zoom", "Signal", "MicrosoftTeams", "GoogleMeet", "Viber",
        "FacebookMessenger", "WeChat", "Line", "Kik", "Snapchat", "Instagram", "Twitter", "Facebook", "LinkedIn", "Reddit",
        "TikTok", "Clubhouse", "Mastodon", "Threads", "BeReal", "Spotify", "AppleMusic", "YouTube", "Netflix", "Hulu",
        "DisneyPlus", "AmazonPrime", "HBOMax", "Twitch", "SoundCloud", "Deezer", "Pandora", "Tidal", "GoogleDrive",
        "GoogleDocs", "Evernote", "Notion", "Trello", "Asana", "Monday", "ClickUp", "Todoist", "OneNote", "Dropbox",
        "PayPal", "Venmo", "CashApp", "Zelle", "GooglePay", "ApplePay", "Stripe", "Robinhood", "Revolut", "Wise"
    ]
    random_name_choice = random.choice(names)
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    output_filename = f"{random_name_choice}_{random_suffix}.py"

    temp_folder = "temp"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)

    output_path = os.path.join(temp_folder, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    print(f"Spoof complete! Generated {output_path}")
    print("Running the generated file...")
    subprocess.run([sys.executable, output_path])

if __name__ == "__main__":
    main()
