import os

DATA_DIR = "./data"
print(f"Checking {os.path.abspath(DATA_DIR)}...")

if not os.path.exists(DATA_DIR):
    print("ERROR: Data directory does not exist!")
else:
    print("Data directory exists.")
    count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        print(f"Scanning: {root}")
        for f in files:
            if f.lower().endswith('.png'):
                count += 1
                if count <= 5:
                    print(f"  Found: {f}")
    
    print(f"Total PNGs found: {count}")
