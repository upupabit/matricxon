"""Built-in chat templates and template markers ChatTemplatePromptBuilder uses (chat_template.py):
templates for GGUFs that ship none, Llama 3's compact form, and the structured-content marker."""

# Built-in templates for models whose GGUF ships no `tokenizer.chat_template`, keyed by
# `general.name`: (template, always add BOS). moondream2's reference format (BOS first, system
# messages skipped: they only confuse a 1.9B model); the old Mistral fallback's first generated
# token was EOS here (empty reply).
MOONDREAM_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}{{ m.image_markers + '\n\nQuestion: ' + m.text }}"
    "{% elif m.role == 'assistant' %}{{ '\n\nAnswer: ' + m.text }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n\nAnswer:' }}{% endif %}"
)
BUILTIN_TEMPLATES = {"moondream2": (MOONDREAM_TEMPLATE, True)}

# Llama 3.x's template always opens with a system block, even without a system message/tools;
# Ollama's llama3 template writes one only with a system message or tools (for "Say hi." Ollama
# prefilled 13 tokens, Matricxon 40: most of a short reply's CPU latency). Used when
# there are no tools; with tools, the model's template renders them.
LLAMA3_COMPACT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{{ '<|start_header_id|>' + ('ipython' if m.role == 'tool' else m.role)"
    " + '<|end_header_id|>\n\n' + m.content + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
LLAMA3_TEMPLATE_MARKERS = ("<|start_header_id|>", "Cutting Knowledge Date")

# Marker (SmolVLM2's template: `message['content'][0]['type']`) for a template expecting `content`
# as a list of `{"type":...}` parts (HF's multi-modal convention), not a flat string. a flat
# string crashes on empty content ("str object has no element 0") or renders empty on non-empty
# content (Jinja `Undefined` never matches).
STRUCTURED_CONTENT_MARKERS = ("content'][0]", 'content"][0]')
