"""
Shared helpers used across multiple nodes.
"""


def extract_text(content) -> str:
    """
    Newer versions of langchain-google-genai sometimes return
    response.content as a list of content blocks instead of a plain
    string, depending on the model. This normalizes either shape into a
    single string, so every node can safely call .strip() on the result
    without caring which shape the underlying library returned.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)