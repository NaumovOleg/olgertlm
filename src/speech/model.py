from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")


def generate_response(user_input):

    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
