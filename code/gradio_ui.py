"""a polished gradio chat interface for stock-lens

built on gr.ChatInterface for robust message handling, extended with a
copy-entire-conversation control, refined styling, and clear hover states
"""

import inspect

DESCRIPTION = ("Ask in plain English — predictions, confidence, feature "
               "explanations, recent news, sector rankings, or top stocks. "
               "Dataset covers 2010–2016; add the word \"live\" for a current "
               "quote, e.g. \"Will TSLA go up tomorrow? live\".")

EXAMPLES = [
    "Which stocks have the highest predicted probability of a gain tomorrow?",
    "Which sector should I buy into for gains next week?",
    "Is now a good time to buy into the Energy sector?",
    "Why is XOM predicted the way it is?",
    "How confident are you about Microsoft?",
    "What's the latest news on Apple?",
]

# refined styling: readable bubbles, clear button hover/active states,
# a calm indigo accent, and a tidy transcript panel
CSS = """
.gradio-container {max-width: 920px !important; margin: auto;}
#sl-title {text-align:center; font-weight:700; font-size:1.7rem;
           margin-bottom:2px;}
#sl-sub {text-align:center; color:#6b7280; font-size:0.92rem;
         margin-top:0; margin-bottom:10px;}
button {transition: background-color .15s ease, transform .05s ease,
        box-shadow .15s ease !important;}
button:hover {filter: brightness(1.05);
              box-shadow: 0 2px 8px rgba(79,70,229,.18) !important;}
button:active {transform: translateY(1px);}
#sl-copyall {min-width:210px;}
#sl-copyall.copied {background:#059669 !important; color:#fff !important;}
#sl-transcript textarea {font-family: ui-monospace, SFMono-Regular, Menlo,
                         monospace; font-size:0.82rem; line-height:1.35;}
footer {visibility:hidden;}
"""

COPY_JS = """
(text) => {
  if (navigator.clipboard && text) {
    navigator.clipboard.writeText(text);
  }
  const b = document.getElementById('sl-copyall');
  if (b) {
    const btn = b.querySelector('button') || b;
    const old = btn.textContent;
    btn.textContent = 'Copied to clipboard';
    b.classList.add('copied');
    setTimeout(() => { btn.textContent = old; b.classList.remove('copied'); },
               1400);
  }
  return text;
}
"""


def build_interface(answer_fn):
    # wrapping the plain answer() in gradio's chat UI plus extras
    import gradio as gr

    def respond(message, history):
        message = (message or "").strip()
        if not message:
            return "Ask me about a stock, its news, a sector, or the top movers."
        try:
            reply = answer_fn(message)
        except Exception as e:
            reply = f"Sorry, something went wrong answering that: {e}"
        return "" if reply is None else str(reply)

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    ci_params = inspect.signature(gr.ChatInterface).parameters

    with gr.Blocks(theme=theme, css=CSS, title="stock-lens") as demo:
        gr.Markdown("stock-lens", elem_id="sl-title")
        gr.Markdown(DESCRIPTION, elem_id="sl-sub")

        ci_kwargs = {"fn": respond, "examples": EXAMPLES}
        if "type" in ci_params:
            ci_kwargs["type"] = "messages"
        if "cache_examples" in ci_params:
            ci_kwargs["cache_examples"] = False
        chat = gr.ChatInterface(**ci_kwargs)

        # locate the ChatInterface's own chatbot component to read its state
        chatbot_comp = None
        for comp in demo.blocks.values():
            if isinstance(comp, gr.Chatbot):
                chatbot_comp = comp
                break

        with gr.Row():
            copy_all = gr.Button("Copy entire conversation",
                                 elem_id="sl-copyall", size="sm",
                                 variant="secondary")
        transcript = gr.Textbox(visible=False, elem_id="sl-hidden")

        def build_transcript(history):
            # flattening the conversation into copyable plain text
            if not history:
                return ""
            lines = []
            for turn in history:
                if isinstance(turn, dict):
                    who = "You" if turn.get("role") == "user" else "stock-lens"
                    lines.append(f"{who}: {turn.get('content', '')}")
                elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                    lines.append(f"You: {turn[0]}")
                    lines.append(f"stock-lens: {turn[1]}")
            return "\n\n".join(lines)

        if chatbot_comp is not None:
            # build transcript text, then copy it client-side with feedback
            copy_all.click(build_transcript, chatbot_comp, transcript) \
                .then(None, transcript, None, js=COPY_JS)

    return demo
