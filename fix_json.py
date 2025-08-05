import json

def fix_jsonl_file(filename):
    fixed_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    # Try to parse the line as JSON
                    data = json.loads(line)
                    fixed_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error on line {line_num}: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
    
    # Write the fixed data back
    with open(filename + '.fixed', 'w', encoding='utf-8') as f:
        for item in fixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Fixed {len(fixed_data)} entries in {filename}")

# Fix both files
fix_jsonl_file('enhanced_corpus.jsonl')
fix_jsonl_file('qna_corpus.jsonl') 