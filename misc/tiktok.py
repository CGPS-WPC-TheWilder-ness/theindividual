# Install tiktoken if you haven't already:
# pip install tiktoken

import tiktoken

def count_tokens(text):
    # Load the GPT-4 model encoding (GPT-4 uses 'gpt-4' or 'gpt-4-0613' as the encoding)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    # Encode the text
    tokenized_text = encoding.encode(text)
    
    # Return the number of tokens
    return len(tokenized_text)

if __name__ == "__main__":
    # Hardcoded text for testing
    test_text = """
The core peace bias is a set of guiding principles designed to steer us towards a more peaceful and harmonious existence. Criticizing these biases can be a valuable exercise to understand their strengths and weaknesses better. Let's delve into some potential criticisms and their implications:

Nonviolence > Violence: While nonviolence is a noble goal, critics might argue that it is not always practical. In situations where immediate self-defense is necessary, nonviolence may not be a viable option. For example, if a community is under attack, the principle of nonviolence might hinder their ability to protect themselves effectively.

Local > Global: This bias emphasizes the importance of fostering local connections before global ones. Critics might argue that in an increasingly interconnected world, focusing too much on local issues could lead to neglecting global challenges like climate change or international conflicts. However, favoring local connections can promote network growth by triggering explosive percolation events, which can eventually unify world piece computers on a global level.

Compassion > Sympathy: Compassion involves a deeper level of understanding and action compared to sympathy. Critics might argue that always prioritizing compassion can be emotionally exhausting and may lead to burnout. For instance, constantly engaging in compassionate acts without adequate self-care can deplete one's emotional resources.

Commitment > Non-commitment: While commitment is essential for achieving long-term goals, critics might point out that it can also lead to rigidity. In some cases, being overly committed to a particular course of action can prevent individuals or groups from adapting to new information or changing circumstances.

Tolerance > Intolerance: Tolerance is crucial for peaceful coexistence, but critics might argue that it can sometimes lead to the acceptance of harmful behaviors. For example, tolerating discriminatory practices under the guise of respecting cultural differences can perpetuate injustice.

Trust > Distrust: Trust is fundamental for building strong relationships, but critics might argue that it can make individuals or groups vulnerable to exploitation. For instance, placing too much trust in unverified sources of information can lead to misinformation and manipulation.

Democracy > Autocracy: Democracy promotes inclusivity and shared decision-making, but critics might argue that it can be inefficient. In situations requiring swift action, the democratic process can be slow and cumbersome, potentially hindering timely responses to crises.

Despite these criticisms, the core peace biases are designed to guide us towards a more peaceful and inclusive world. By considering life as a random walk, we can understand that without weights such as the core peace bias, we may never truly progress. By deliberately choosing our biases, we can drift towards a more peaceful direction over time.

Ultimately, the core peace biases are part of The Human Imperative, which guides the operation of world piece computers. These biases help us make deliberate choices that favor peace, inclusion, acceptance, tolerance, neutrality, compassion, commitment, nonviolence, and nondisruption. By critically examining these biases, we can refine and strengthen our approach to building a universal piece computer and achieving global peace."""

# Count and print the number of tokens
    print(f"Number of tokens: {count_tokens(test_text)}")
