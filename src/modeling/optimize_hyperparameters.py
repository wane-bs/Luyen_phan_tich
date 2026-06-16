import os
import json
import ast
import re
import pandas as pd
import numpy as np

def extract_config(raw_content):
    brace_positions = []
    for m in re.finditer(r'\{', raw_content):
        brace_positions.append(('{', m.start()))
    for m in re.finditer(r'\}', raw_content):
        brace_positions.append(('}', m.start()))
        
    brace_positions.sort(key=lambda x: x[1])
    candidates = []
    
    for i in range(len(brace_positions)):
        if brace_positions[i][0] == '{':
            for j in range(i + 1, len(brace_positions)):
                if brace_positions[j][0] == '}':
                    start_idx = brace_positions[i][1]
                    end_idx = brace_positions[j][1]
                    candidate = raw_content[start_idx:end_idx+1]
                    candidates.append(candidate)
                    
    required_keys = [
        "credit_score_min", "credit_score_max", 
        "current_age_min", "current_age_max",
        "max_depth", "min_child_weight", 
        "reg_alpha", "reg_lambda"
    ]
    
    # Try all candidate substrings
    for cand in candidates:
        cand_str = cand.strip()
        parsed_dict = None
        
        # 1. Try pure JSON
        try:
            parsed_dict = json.loads(cand_str)
        except Exception:
            pass
            
        # 2. Try JSON with single-to-double quote replacement
        if parsed_dict is None:
            try:
                parsed_dict = json.loads(cand_str.replace("'", '"'))
            except Exception:
                pass
                
        # 3. Try Python dict literal evaluation
        if parsed_dict is None:
            try:
                parsed_dict = ast.literal_eval(cand_str)
            except Exception:
                pass
                
        # If successfully parsed, sanitize keys and validate
        if isinstance(parsed_dict, dict):
            # Strip whitespace and lowercase keys
            sanitized = {}
            for k, v in parsed_dict.items():
                if isinstance(k, str):
                    sanitized[k.strip().lower()] = v
                else:
                    sanitized[k] = v
            
            # Check if all required keys are present
            if all(k in sanitized for k in required_keys):
                try:
                    # Enforce business rules and type safety
                    config = {
                        "credit_score_min": max(300.0, float(sanitized["credit_score_min"])),
                        "credit_score_max": min(850.0, float(sanitized["credit_score_max"])),
                        "current_age_min": max(18.0, float(sanitized["current_age_min"])),
                        "current_age_max": min(100.0, float(sanitized["current_age_max"])),
                        "max_depth": int(sanitized["max_depth"]),
                        "min_child_weight": float(sanitized["min_child_weight"]),
                        "reg_alpha": float(sanitized["reg_alpha"]),
                        "reg_lambda": float(sanitized["reg_lambda"])
                    }
                    return config
                except Exception as parse_err:
                    print(f"Error casting types or enforcing rules on candidate: {parse_err}")
                    pass
            
    raise ValueError("No valid configuration block found in LLM response.")

def run_pre_training_optimization():
    print("Executing pre-training optimization using qwen2.5-math:1.5b...")
    
    csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Please run src/data_pipeline/feature_engineering.py first.")
        # Create a fallback config
        write_fallback()
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading feature matrix: {e}")
        write_fallback()
        return
    
    stats = {}
    for col in ['credit_score', 'current_age']:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        else:
            print(f"Warning: Column {col} not found in features matrix.")
            stats[col] = {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 100.0}
        
    prompt = f"""
    You are an expert mathematical optimization system for risk scoring models. 
    Analyze the following feature statistics from our credit underwriting database:
    {json.dumps(stats, indent=2)}
    
    Task:
    1. Calculate logical clipping thresholds to handle Gaussian noise without losing critical feature distributions.
    2. Propose optimized regularized parameters for XGBoost (max_depth, min_child_weight, reg_alpha, reg_lambda).
    
    Your output MUST be a single raw JSON block with the following keys, and nothing else (no reasoning, no extra text):
    {{
        "credit_score_min": <float>,
        "credit_score_max": <float>,
        "current_age_min": <float>,
        "current_age_max": <float>,
        "max_depth": <int>,
        "min_child_weight": <float>,
        "reg_alpha": <float>,
        "reg_lambda": <float>
    }}
    """
    
    try:
        from ollama import chat
        response = chat(
            model='mightykatun/qwen2.5-math:1.5b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={
                'temperature': 0.2,
                'num_ctx': 4096
            }
        )
        raw_content = response.message.content.strip()
        print("Raw response from mightykatun/qwen2.5-math:1.5b:")
        print(raw_content)
        print("-" * 40)
        
        config = extract_config(raw_content)
        
        # Validate critical fields
        required_keys = [
            "credit_score_min", "credit_score_max", 
            "current_age_min", "current_age_max",
            "max_depth", "min_child_weight", 
            "reg_alpha", "reg_lambda"
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing key in LLM output: {key}")
                
        # Ensure correct types
        config["max_depth"] = int(config["max_depth"])
        config["min_child_weight"] = float(config["min_child_weight"])
        config["reg_alpha"] = float(config["reg_alpha"])
        config["reg_lambda"] = float(config["reg_lambda"])
        
        os.makedirs(os.path.join("data", "configs"), exist_ok=True)
        config_path = os.path.join("data", "configs", "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Generated {config_path} successfully via LLM optimization.")
        print(json.dumps(config, indent=2))
        
    except Exception as e:
        print("Fallback to standard mathematical rules due to error:", e)
        write_fallback()

def write_fallback():
    fallback = {
        "credit_score_min": 300.0,
        "credit_score_max": 850.0,
        "current_age_min": 18.0,
        "current_age_max": 100.0,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "reg_alpha": 1.5,
        "reg_lambda": 3.0
    }
    config_path = os.path.join("data", "configs", "model_config.json")
    os.makedirs(os.path.join("data", "configs"), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(fallback, f, indent=4)
    print(f"Fallback configuration written to {config_path}.")

if __name__ == "__main__":
    run_pre_training_optimization()
