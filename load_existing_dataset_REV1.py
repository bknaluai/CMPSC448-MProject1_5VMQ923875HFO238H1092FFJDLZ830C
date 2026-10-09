import pandas as pd 
from datasets import load_dataset 

print("Downloading LMSYS Chatbot Arena dataset...") 
# Load dataset 
ds = load_dataset("agie-ai/lmsys-chatbot_arena_conversations", split="train") 

def map_to_family(model_name): 
    """Group 100+ model variants into top balanced LLM families.""" 
    if not isinstance(model_name, str): 
        return "Other" 
    name = model_name.lower() 
    if any(k in name for k in ['gpt', 'openai']): 
        return 'OpenAI (GPT)' 
    elif any(k in name for k in ['claude', 'anthropic']): 
        return 'Anthropic (Claude)' 
    elif any(k in name for k in ['llama', 'vicuna', 'alpaca', 'guanaco']): 
        return 'Meta (Llama)' 
    elif any(k in name for k in ['palm', 'gemini', 'bard']): 
        return 'Google (Gemini)' 
    elif any(k in name for k in ['mistral', 'zephyr', 'mixtral', 'qwen']): 
        return 'Mistral/Qwen' 
    return 'Other' 

def infer_task_type(prompt): 
    """Categorize prompt domain for RQ3 cross-domain generalization.""" 
    p = str(prompt).lower() 
    if any(k in p for k in ['code', 'python', 'function', 'bug', 'html', 'java', 'c++']): 
        return 'coding' 
    elif any(k in p for k in ['write a story', 'poem', 'essay', 'creative', 'roleplay', 'song']): 
        return 'creative_writing' 
    elif any(k in p for k in ['math', 'calculate', 'prove', 'solve', 'equation']): 
        return 'math' 
    return 'general_qa' 

formatted_data = [] 

# Scan up to 15,000 interactions to gather sufficient samples per family
for _, row in ds.to_pandas().head(15000).iterrows(): 
    conv_a = row['conversation_a'] 
    conv_b = row['conversation_b'] 
    
    if len(conv_a) >= 2 and len(conv_b) >= 2: 
        prompt = conv_a[0]['content'] 
        task = infer_task_type(prompt) 

        fam_a = map_to_family(row['model_a']) 
        if fam_a != 'Other': 
            formatted_data.append({ 
                "LLM_name": fam_a, 
                "raw_model": row['model_a'], 
                "LLM_Input": prompt, 
                "LLM_output": conv_a[1]['content'], 
                "task_type": task 
            }) 
            
        fam_b = map_to_family(row['model_b']) 
        if fam_b != 'Other': 
            formatted_data.append({ 
                "LLM_name": fam_b, 
                "raw_model": row['model_b'], 
                "LLM_Input": prompt, 
                "LLM_output": conv_b[1]['content'], 
                "task_type": task 
            }) 

df = pd.DataFrame(formatted_data) 

# Select top 3 families with at least 500 samples each
family_counts = df['LLM_name'].value_counts()
valid_families = family_counts[family_counts >= 500].head(3).index.tolist()

if len(valid_families) < 3:
    print(f"Warning: Only found {len(valid_families)} families with >= 500 samples. Using top available.")
    valid_families = family_counts.head(3).index.tolist()

# Sample exactly 500 rows per selected family
df_balanced = (
    df[df['LLM_name'].isin(valid_families)]
    .groupby('LLM_name')
    .head(500)
    .reset_index(drop=True)
)

df_balanced.to_csv("llm_dataset.csv", index=False) 

print(f"\nDone! Saved {len(df_balanced)} samples to 'llm_dataset.csv'.") 
print("\nLLM Families Selected:") 
print(df_balanced["LLM_name"].value_counts()) 
print("\nTask Types Breakdown:") 
print(df_balanced["task_type"].value_counts())