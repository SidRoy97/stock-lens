"""a polished gradio chat interface for stock-lens

built on gr.ChatInterface, which manages the chat message format
internally, so the answer function only ever returns a plain string —
this avoids the messages-format pitfalls of hand-built chat Blocks
"""

import inspect

DESCRIPTION = ("Ask in plain English — predictions, confidence, feature "
               "explanations, sector rankings, or top stocks. Data covers "
               "2010–2016; “today” means the latest day in the dataset.")

EXAMPLES = [
    "Which stocks have the highest predicted probability of a gain tomorrow?",
    "Which sector should I buy into for gains next week?",
    "Is now a good time to buy into the Energy sector?",
    "Why is XOM predicted the way it is?",
    "How confident are you about Microsoft?",
    "Should I sell or hold my Apple stock?",
]


def build_interface(answer_fn):
    # wrapping the plain answer() in gradio's purpose-built chat UI
    import gradio as gr

    def respond(message, history):
        # ChatInterface hands us the raw string and manages formatting;
        # any input is valid and we always return a string
        message = (message or "").strip()
        if not message:
            return "Ask me about a stock, a sector, or the top movers."
        try:
            reply = answer_fn(message)
        except Exception as e:
            reply = f"Sorry, something went wrong answering that: {e}"
        return "" if reply is None else str(reply)

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    params = inspect.signature(gr.ChatInterface).parameters
    kwargs = {"fn": respond, "title": "🔍 stock-lens",
              "description": DESCRIPTION, "examples": EXAMPLES}
    # only pass kwargs this gradio version actually supports
    if "theme" in params:
        kwargs["theme"] = theme
    if "type" in params:
        kwargs["type"] = "messages"
    if "cache_examples" in params:
        kwargs["cache_examples"] = False

    try:
        return gr.ChatInterface(**kwargs)
    except TypeError:
        # last-resort minimal signature for older/newer gradio
        return gr.ChatInterface(fn=respond, title="🔍 stock-lens",
                                description=DESCRIPTION)
