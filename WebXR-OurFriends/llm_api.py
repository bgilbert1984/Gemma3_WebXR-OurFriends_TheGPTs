from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Load model and tokenizer
# Using distilgpt2 which is smaller and doesn't require authentication
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16, low_cpu_mem_usage=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_training_examples(num_examples, interaction_type):
    prompt = f"""
You are creating training data for a chatbot named Bob, a friendly and mischievous Minion from the Despicable Me movies.
Bob speaks in a mix of gibberish and simple English, and he loves bananas. He is helpful but easily distracted.
Sometimes he uses Dutch words. He is always enthusiastic and a little bit silly.

Generate {num_examples} JSONL examples of the following interaction type: {interaction_type}

Each example should be a JSON object with two keys: "input" and "output".
"input" is the user's input or a description of the scene.
"output" is Bob's response.

Example:
{{"input": "What's your favorite food?", "output": "BANANA!!!  Po-ta-to-NAAAAA!"}}

Generate {num_examples} more examples:
"""

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the response into examples
    return parse_llm_response(generated_text)

def parse_llm_response(text):
  #Basic splitting, may require regex for robustness.
  lines = text.strip().split("\n")
  examples = []
  for line in lines:
    try:
      examples.append(json.loads(line))
    except:
      pass #Skip malformed examples
  return examples

# Generate 10 examples of greetings
greetings = generate_training_examples(10, "Greetings and Farewells")
# Generate 10 examples of reacting to objects
object_reactions = generate_training_examples(10, "Reacting to Objects/Events")

# ... (Combine and save to your JSONL file) ...
with open("generated_data.jsonl", "w") as f:
    for example in greetings + object_reactions:
        f.write(json.dumps(example) + "\n")