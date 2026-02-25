"""
Activation Extraction

Runs GPT-2 through TransformerLens, extracting the complete residual stream
at all 13 checkpoints (pre-layer-0 through post-layer-12) for every token
position. Saves to HDF5 for efficient memory-mapped access during training.

This is the "neural tap" — read-only access to the executor's brain activity.
"""

import numpy as np
import h5py
import torch
from tqdm import tqdm
from pathlib import Path

from . import config


# ── Curated text sequences ─────────────────────────────────────────────

GARDEN_PATH_SENTENCES = [
    "The horse raced past the barn fell down suddenly without warning.",
    "The old man the boat in the harbor every weekend without fail.",
    "The complex houses married and single soldiers and their families.",
    "The cotton clothing is made of grows in Mississippi fields.",
    "The girl told the story cried after hearing the ending.",
    "The man who hunts ducks out on weekends to go fishing.",
    "The raft floated down the river sank after hitting a rock.",
    "Fat people eat accumulates in their bodies over many years.",
    "The prime number few can be computed without significant effort.",
    "The dog that bit the cat scratched ran away into the woods.",
    "Time flies like an arrow but fruit flies like a banana.",
    "The horse that was raced past the barn fell down hard.",
    "Have the students who failed the exam take the supplementary test.",
    "The government plans to raise taxes were defeated by the opposition.",
    "The patient persuaded the doctor that he was having trouble with.",
    "The boat floated down the river sank beneath the muddy water.",
    "I convinced her children are noisy and should be kept outside.",
    "The defendant examined by the lawyer turned out to be guilty.",
    "The chicken is ready to eat its food in the morning.",
    "We painted the wall with cracks to hide the damage underneath.",
    "The man returned to his house was happy to see everything intact.",
    "While the woman was sewing the dress fell off the table.",
    "The emergency room treated patients with injuries from the storm.",
    "She told her baby could not swim yet in the pool.",
    "The girl found in the abandoned building was taken to safety.",
    "The coach smiled at the player tossed the frisbee across.",
    "Without her contributions would be impossible to the organization.",
    "The landlord painted the walls with flowers to brighten the room.",
    "After the boy left the room was quiet for hours.",
    "The old train the young on the ways of the world.",
]

DOMAIN_SWITCH_TEMPLATES = [
    # Legal -> Poetry
    ("The defendant hereby agrees to the terms and conditions set forth in this binding agreement, "
     "including but not limited to the obligations described in Section 4.2 of the contract. "
     "Roses are red, violets are blue, the moonlight dances on morning dew."),
    # Code -> Prose
    ("def fibonacci(n): if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2). "
     "The sunset painted the sky in brilliant shades of orange and purple, "
     "casting long shadows across the quiet village."),
    # Math -> Narrative
    ("Consider the integral of x squared dx from 0 to infinity. The convergence "
     "of this series depends on the ratio test. Meanwhile, in the small town of "
     "Millbrook, a young girl discovered a mysterious letter in her grandmother's attic."),
    # Science -> Casual
    ("The mitochondria is the powerhouse of the cell, responsible for ATP production "
     "through oxidative phosphorylation in the electron transport chain. "
     "Anyway dude, wanna grab pizza tonight? I'm totally starving lol."),
    # History -> Technical
    ("In 1776, the Continental Congress declared independence from Great Britain, "
     "marking a pivotal moment in world history. The HTTP/2 protocol uses "
     "multiplexing to send multiple requests over a single TCP connection."),
    # News -> Fiction
    ("BREAKING: Markets tumble as inflation data exceeds expectations. The Federal "
     "Reserve is expected to announce rate decisions tomorrow. "
     "The dragon unfurled its wings and took flight over the burning city."),
    # Recipe -> Philosophy
    ("Preheat the oven to 375 degrees. Mix flour, sugar, and butter until crumbly. "
     "Add eggs one at a time. What is the nature of consciousness? "
     "Is there a self that persists through time, or merely a stream?"),
    # Medical -> Sports
    ("The patient presents with acute myocardial infarction, elevated troponin levels, "
     "and ST-segment elevation in leads II, III, and aVF. "
     "And the crowd goes wild as the striker scores in the 90th minute!"),
    # Academic -> Slang
    ("The epistemological implications of Cartesian dualism have been debated extensively "
     "in contemporary philosophy of mind. Bro that test was absolutely bussin no cap, "
     "I literally aced every single question fr fr."),
    # Business -> Poetry
    ("Q3 earnings exceeded analyst expectations by 12%, driven by strong performance "
     "in the cloud services division. Do not go gentle into that good night, "
     "old age should burn and rave at close of day."),
]

REASONING_PROMPTS = [
    "What is 247 + 389? Let me think step by step: 247 + 389 = 636.",
    "What is 15 * 23? Let me compute: 15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345.",
    "If all dogs are mammals and all mammals are animals, then all dogs are animals.",
    "The sequence 2, 4, 8, 16, 32 follows the pattern of doubling. The next number is 64.",
    "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? No, that's affirming the consequent.",
    "What is the square root of 144? The answer is 12, because 12 * 12 = 144.",
    "A train travels at 60 mph for 2.5 hours. Distance = speed * time = 60 * 2.5 = 150 miles.",
    "If x + 5 = 12, then x = 12 - 5 = 7. Checking: 7 + 5 = 12. Correct.",
    "The binary number 1010 equals 1*8 + 0*4 + 1*2 + 0*1 = 10 in decimal.",
    "def reverse(s): return s[::-1]. For example, reverse('hello') returns 'olleh'.",
    "In a group of 30 students, 18 play soccer and 15 play basketball. At least 18 + 15 - 30 = 3 play both.",
    "The factorial of 5 is 5! = 5 * 4 * 3 * 2 * 1 = 120.",
    "If P implies Q, and not Q, then not P. This is modus tollens.",
    "What is 1000 - 387? Let me compute: 1000 - 387 = 613.",
    "The sum of angles in a triangle is 180 degrees. If two angles are 60 and 70, the third is 50.",
    "Convert 25 Celsius to Fahrenheit: F = 25 * 9/5 + 32 = 45 + 32 = 77 F.",
    "The GCD of 48 and 36: 48 = 1*36 + 12, 36 = 3*12 + 0. So GCD is 12.",
    "Is 97 prime? Check divisibility: not by 2, 3, 5, 7. Since sqrt(97) < 10, yes it is prime.",
    "A list sorted in ascending order: [3, 1, 4, 1, 5] -> [1, 1, 3, 4, 5].",
    "If a function f(x) = x^2, then f'(x) = 2x. At x=3, the slope is 6.",
]


def _get_pile_texts(n_sequences: int, seq_len: int, tokenizer) -> list[str]:
    """Get diverse text sequences for activation extraction.

    Uses a curated set of diverse text types since The Pile requires
    special access. Falls back to generating diverse prompts.
    """
    texts = []

    # Use diverse seed texts covering different domains
    diverse_seeds = [
        # Wikipedia-style
        "The history of mathematics begins with the ancient Babylonians and Egyptians, who developed",
        "Photosynthesis is the process by which green plants and certain other organisms transform",
        "The French Revolution of 1789 was a period of radical political and societal change in France",
        "Quantum mechanics is a fundamental theory in physics that describes the behavior of nature",
        "The human brain contains approximately 86 billion neurons, each connected to thousands",
        "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused",
        "The Renaissance was a cultural movement that profoundly affected European intellectual life",
        "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil",
        "The theory of general relativity, published by Albert Einstein in 1915, describes gravity",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems",

        # Code
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
        "function mergeSort(arr) {\n    if (arr.length <= 1) return arr;\n    const mid = Math.floor(",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
        "CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    username VARCHAR(255) UNIQUE NOT NULL,",
        "pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let mut low = 0;",

        # Literary
        "It was the best of times, it was the worst of times, it was the age of wisdom,",
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune",
        "The sun shone, having no alternative, on the nothing new. Murphy sat out of it, as though",

        # Scientific papers
        "Abstract: We present a novel approach to protein folding prediction using deep learning",
        "The results demonstrate a statistically significant correlation (p < 0.001) between",
        "Recent advances in large language models have shown remarkable capabilities in natural",
        "Methods: We collected samples from 150 participants aged 18-65, randomly assigned to",
        "Our findings suggest that the observed phenomenon can be explained by a combination of",

        # News/journalism
        "WASHINGTON - The Senate voted today to approve the new infrastructure bill, marking a major",
        "Scientists at CERN announced the discovery of a previously unknown subatomic particle that",
        "The global economy showed signs of recovery in the third quarter, with GDP growth exceeding",
        "In a landmark ruling, the Supreme Court decided that the constitutional protections extend",
        "Researchers at Stanford University have developed a new AI system capable of diagnosing",

        # Conversational
        "So basically what happened was, I was walking down the street minding my own business when",
        "Hey everyone, welcome back to the channel! Today we're going to be talking about something",
        "Dear diary, today was probably the worst day of my entire life. First, my alarm didn't go",
        "Okay so here's the thing about machine learning that nobody really talks about: most of the",
        "I think the most important lesson I've learned in my career is that communication matters",

        # Technical documentation
        "To install the package, run: pip install transformer-lens. Requirements: Python 3.8+,",
        "The API endpoint accepts POST requests with a JSON body containing the following fields:",
        "Configuration: Set the environment variable CUDA_VISIBLE_DEVICES to select GPU devices.",
        "Error handling: When the server returns a 429 status code, implement exponential backoff",
        "The database schema consists of three primary tables: users, sessions, and transactions.",

        # Philosophy
        "The hard problem of consciousness asks why and how physical processes in the brain give rise",
        "Descartes argued that the mind and body are fundamentally different substances, a position",
        "The Chinese Room argument, proposed by John Searle, challenges the notion that a computer",
        "What does it mean to be conscious? This question has puzzled philosophers and scientists",
        "The problem of other minds asks how we can know that other beings have conscious experiences",

        # Math
        "Theorem: For any prime p and any integer a not divisible by p, we have a^(p-1) is congruent",
        "Let f be a continuous function on the closed interval [a,b]. Then there exists a point c in",
        "The Riemann hypothesis states that all non-trivial zeros of the Riemann zeta function have",
        "Consider the vector space V over the field F. A linear transformation T: V -> V is called",
        "The fundamental theorem of calculus states that if F is an antiderivative of f on [a,b],",
    ]

    # Repeat and vary seeds to fill n_sequences
    import random
    rng = random.Random(42)
    for i in range(n_sequences):
        seed = diverse_seeds[i % len(diverse_seeds)]
        # Add some variation
        if i >= len(diverse_seeds):
            # Extend with continuations
            continuations = [
                " Furthermore, ", " Additionally, ", " In contrast, ",
                " However, ", " Moreover, ", " Subsequently, ",
                " As a result, ", " Nevertheless, ", " Consequently, ",
                " In particular, ",
            ]
            seed = seed + rng.choice(continuations)
        texts.append(seed)

    return texts


def extract_activations(
    model_name: str = None,
    output_path: str = None,
    n_sequences: int = None,
    seq_len: int = None,
    device: torch.device = None,
    save_attention: bool = True,
):
    """Extract residual stream activations from GPT-2 via TransformerLens.

    For each sequence:
    - Runs the model with caching enabled
    - Extracts residual stream at all 13 checkpoints: [pre, post_0, ..., post_11]
    - Optionally extracts attention patterns
    - Saves to HDF5

    Args:
        model_name: HuggingFace model ID (default: config.EXECUTOR_MODEL)
        output_path: Where to save HDF5 (default: config.ACTIVATIONS_PATH)
        n_sequences: Number of sequences (default: config.N_SEQUENCES)
        seq_len: Tokens per sequence (default: config.SEQ_LEN)
        device: Torch device (default: config.DEVICE)
        save_attention: Whether to save attention patterns
    """
    import transformer_lens

    model_name = model_name or config.EXECUTOR_MODEL
    output_path = Path(output_path or config.ACTIVATIONS_PATH)
    n_sequences = n_sequences or config.N_SEQUENCES
    seq_len = seq_len or config.SEQ_LEN
    device = device or config.DEVICE

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} via TransformerLens...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        device=str(device) if device.type != "mps" else "cpu",
    )
    # MPS: TransformerLens may not fully support MPS, so run extraction on CPU
    # and let training use MPS
    if device.type == "mps":
        print("  Note: Running extraction on CPU (TransformerLens + MPS compatibility)")
        model = model.to("cpu")
        extract_device = torch.device("cpu")
    else:
        extract_device = device

    model.eval()
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.cfg.n_layers  # 12 for gpt2
    d_model = model.cfg.d_model    # 768 for gpt2
    n_heads = model.cfg.n_heads    # 12 for gpt2

    print(f"Model: {model_name} | Layers: {n_layers} | d_model: {d_model} | Heads: {n_heads}")
    print(f"Extracting {n_sequences} sequences of length {seq_len}")

    # ── Prepare text sequences ──────────────────────────────────────
    pile_texts = _get_pile_texts(config.N_PILE_SEQUENCES, seq_len, tokenizer)
    garden_texts = GARDEN_PATH_SENTENCES
    domain_texts = DOMAIN_SWITCH_TEMPLATES
    reasoning_texts = REASONING_PROMPTS

    all_texts = pile_texts[:config.N_PILE_SEQUENCES]

    # Pad curated texts to target count by cycling
    while len(garden_texts) < config.N_GARDEN_PATH:
        garden_texts = garden_texts + garden_texts
    garden_texts = garden_texts[:config.N_GARDEN_PATH]

    while len(domain_texts) < config.N_DOMAIN_SWITCH:
        domain_texts = domain_texts + domain_texts
    domain_texts = domain_texts[:config.N_DOMAIN_SWITCH]

    while len(reasoning_texts) < config.N_REASONING:
        reasoning_texts = reasoning_texts + reasoning_texts
    reasoning_texts = reasoning_texts[:config.N_REASONING]

    all_texts.extend(garden_texts)
    all_texts.extend(domain_texts)
    all_texts.extend(reasoning_texts)
    all_texts = all_texts[:n_sequences]

    # ── Create HDF5 file ────────────────────────────────────────────
    n_checkpoints = n_layers + 1  # 13 for gpt2 (pre + post each layer)

    with h5py.File(output_path, 'w') as f:
        # Residual stream: (n_sequences, n_checkpoints, seq_len, d_model)
        resid_ds = f.create_dataset(
            'residual_stream',
            shape=(n_sequences, n_checkpoints, seq_len, d_model),
            dtype='float32',
            chunks=(1, n_checkpoints, seq_len, d_model),
            compression='gzip',
            compression_opts=4,
        )

        if save_attention:
            # Attention patterns: (n_sequences, n_layers, n_heads, seq_len, seq_len)
            # This is large — we'll store it but it's optional for training
            attn_ds = f.create_dataset(
                'attention_patterns',
                shape=(n_sequences, n_layers, n_heads, seq_len, seq_len),
                dtype='float16',  # Half precision to save space
                chunks=(1, n_layers, n_heads, seq_len, seq_len),
                compression='gzip',
                compression_opts=4,
            )

        # Token IDs for reference
        tokens_ds = f.create_dataset(
            'token_ids',
            shape=(n_sequences, seq_len),
            dtype='int32',
        )

        # Sequence type labels
        seq_types = []
        for i in range(n_sequences):
            if i < config.N_PILE_SEQUENCES:
                seq_types.append('pile')
            elif i < config.N_PILE_SEQUENCES + config.N_GARDEN_PATH:
                seq_types.append('garden_path')
            elif i < config.N_PILE_SEQUENCES + config.N_GARDEN_PATH + config.N_DOMAIN_SWITCH:
                seq_types.append('domain_switch')
            else:
                seq_types.append('reasoning')
        f.create_dataset('sequence_types', data=np.array(seq_types, dtype='S20'))

        # Metadata
        f.attrs['model_name'] = model_name
        f.attrs['n_sequences'] = n_sequences
        f.attrs['seq_len'] = seq_len
        f.attrs['n_layers'] = n_layers
        f.attrs['n_checkpoints'] = n_checkpoints
        f.attrs['d_model'] = d_model
        f.attrs['n_heads'] = n_heads

        # ── Extract activations ─────────────────────────────────────
        print(f"\nExtracting activations to {output_path}...")

        for i, text in enumerate(tqdm(all_texts, desc="Extracting")):
            # Tokenize
            encoded = tokenizer(text, return_tensors='pt',
                                max_length=seq_len,
                                truncation=True,
                                padding='max_length')
            tokens = encoded['input_ids'][:, :seq_len].to(extract_device)

            if tokens.shape[1] < seq_len:
                # Pad if shorter
                pad = torch.full(
                    (1, seq_len - tokens.shape[1]),
                    tokenizer.pad_token_id,
                    dtype=tokens.dtype,
                    device=extract_device,
                )
                tokens = torch.cat([tokens, pad], dim=1)

            # Run with cache
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: (
                        'hook_resid_pre' in name or
                        'hook_resid_post' in name or
                        ('hook_pattern' in name if save_attention else False)
                    ),
                )

            # Extract residual stream at all checkpoints
            # Checkpoint 0: pre-layer-0 (embedding + pos encoding)
            resid = np.zeros((n_checkpoints, seq_len, d_model), dtype=np.float32)

            # Pre-layer 0 residual
            resid[0] = cache['blocks.0.hook_resid_pre'].squeeze(0).cpu().numpy()

            # Post-layer residuals
            for layer in range(n_layers):
                key = f'blocks.{layer}.hook_resid_post'
                resid[layer + 1] = cache[key].squeeze(0).cpu().numpy()

            resid_ds[i] = resid
            tokens_ds[i] = tokens.squeeze(0).cpu().numpy()

            # Extract attention patterns
            if save_attention:
                attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float16)
                for layer in range(n_layers):
                    key = f'blocks.{layer}.attn.hook_pattern'
                    attn[layer] = cache[key].squeeze(0).cpu().numpy().astype(np.float16)
                attn_ds[i] = attn

            # Clear cache to free memory
            del cache

        print(f"\nSaved {n_sequences} sequences to {output_path}")
        print(f"  Residual stream shape: {resid_ds.shape}")
        if save_attention:
            print(f"  Attention patterns shape: {attn_ds.shape}")
        print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


def validate_activations(
    activations_path: str = None,
    model_name: str = None,
    n_check: int = 10,
):
    """Spot-check extracted activations by verifying they reproduce correct logits.

    Loads a few sequences, runs them through GPT-2 again, and checks that the
    final-layer residual stream matches what we stored.
    """
    import transformer_lens

    activations_path = Path(activations_path or config.ACTIVATIONS_PATH)
    model_name = model_name or config.EXECUTOR_MODEL

    print(f"Validating activations from {activations_path}...")

    model = transformer_lens.HookedTransformer.from_pretrained(model_name, device="cpu")
    model.eval()

    with h5py.File(activations_path, 'r') as f:
        n_sequences = f.attrs['n_sequences']
        n_check = min(n_check, n_sequences)

        errors = []
        for i in range(n_check):
            tokens = torch.tensor(f['token_ids'][i]).unsqueeze(0)
            stored_resid = f['residual_stream'][i]  # (n_checkpoints, seq_len, d_model)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: 'hook_resid_post' in name,
                )

            # Compare final layer residual
            final_layer = model.cfg.n_layers - 1
            actual = cache[f'blocks.{final_layer}.hook_resid_post'].squeeze(0).cpu().numpy()
            stored = stored_resid[-1]  # Last checkpoint = post final layer

            error = np.abs(actual - stored).mean()
            errors.append(error)
            del cache

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"  Checked {n_check} sequences")
        print(f"  Mean absolute error: {mean_error:.8f}")
        print(f"  Max absolute error:  {max_error:.8f}")
        print(f"  Validation {'PASSED' if max_error < 1e-4 else 'FAILED'}")

        return mean_error < 1e-4


if __name__ == '__main__':
    extract_activations()
    validate_activations()
