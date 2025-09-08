import os
import json
import xml.etree.ElementTree as ET

# --- Configuration ---
REPO_DIR = 'osbooks-astronomy' # Assumes this folder is in the same directory
OUTPUT_FILENAME = 'astronomy_textbook_github.json'

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Reading from existing repository: '{REPO_DIR}'")

    # Parse the main collection file
    collection_file = os.path.join(REPO_DIR, 'collections', 'astronomy-2e.collection.xml')
    print(f"Parsing collection file: {collection_file}")
    
    tree = ET.parse(collection_file)
    root = tree.getroot()
    
    ns = {'col': 'http://cnx.rice.edu/collxml'}
    
    module_ids = []
    for module in root.findall('.//col:module', ns):
        module_ids.append(module.get('document'))

    print(f"Found {len(module_ids)} module IDs.")
    
    # Extract text from each module's CNXML file
    textbook_data = {}
    modules_dir = os.path.join(REPO_DIR, 'modules')
    
    for module_id in module_ids:
        cnxml_file_path = os.path.join(modules_dir, module_id, 'index.cnxml')
        
        if os.path.exists(cnxml_file_path):
            try:
                module_tree = ET.parse(cnxml_file_path)
                module_root = module_tree.getroot()
                cnxml_ns = {'cnxml': 'http://cnx.rice.edu/cnxml'}
                
                paragraphs = [p.text for p in module_root.findall('.//cnxml:para', cnxml_ns) if p.text]
                
                if paragraphs:
                    full_text = "\n".join(paragraphs)
                    textbook_data[module_id] = full_text
                    print(f"  - Extracted content from {module_id}")
                    
            except ET.ParseError as e:
                print(f"  - Error parsing {cnxml_file_path}: {e}")
        else:
            print(f"  - CNXML file not found for module {module_id}")

    # Save the final data to a JSON file
    if textbook_data:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(textbook_data, f, indent=4)
        print(f"\n✅ Success! All textbook content saved to '{OUTPUT_FILENAME}'")
    else:
        print("\n❌ No data was extracted.")