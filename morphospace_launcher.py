# Define the path to the external script
script_path = "C:\\Users\\caleb\\OneDrive\\Documents\\Obsidian Vault\\projects\\GitHub Projects\\thicc-snail\\snail_morphospace.py"

# Read and execute the external script
try:
    with open(script_path, 'r') as script_file:
        exec(script_file.read())
    print("Script executed successfully.")
except FileNotFoundError:
    print(f"File not found: {script_path}")
except Exception as e:
    print(f"An error occurred: {e}")
