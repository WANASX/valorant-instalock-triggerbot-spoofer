import os
import sys
import subprocess
import random
import string
import shutil

def insert_random_comments(code):
    lines = code.splitlines()
    new_lines = []
    # Add a random comment at the beginning.
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    for line in lines:
        # Randomly insert a comment before some lines.
        if random.random() < 0.3:
            new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits,
                                                             k=random.randint(20, 40))))
        new_lines.append(line)
    # Add a random comment at the end.
    new_lines.append("# " + ''.join(random.choices(string.ascii_letters + string.digits, k=30)))
    return "\n".join(new_lines)

def process_code_with_ui_markers(code):
    """
    This function processes the input code. It will insert random comments into segments 
    that are not wrapped between UI markers.
    """
    lines = code.splitlines(keepends=True)
    result = []
    segment_lines = []
    in_non_insert = False
    for line in lines:
        if "# UI-DO-NOT-OBFUSCATE-START" in line:
            if segment_lines:
                segment = "".join(segment_lines)
                result.append(insert_random_comments(segment))
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
    if segment_lines:
        segment = "".join(segment_lines)
        result.append(insert_random_comments(segment))
    return "".join(result)

def main():
    input_filename = "menu.py"
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found!")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Process the code by inserting random comments.
    final_code = process_code_with_ui_markers(original_code)

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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    print(f"Spoof complete! Generated {output_path}")
    print("Running the generated file...")
    subprocess.run([sys.executable, output_path])

if __name__ == "__main__":
    main()
