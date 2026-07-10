"""a polished gradio chat interface with session history, copy, and re-ask"""


def build_interface(answer_fn):
    # constructing a chat-style UI around the plain answer() function
    import gradio as gr

    CSS = """
    .gradio-container {max-width: 900px !important; margin: auto;}
    #title-row {text-align:center; padding: 4px 0 0 0;}
    #subtitle {text-align:center; color:#6b7280; margin-top:-6px;
               font-size: 0.9rem;}
    .msg-user {background:#eef2ff; border-radius:12px; padding:10px 14px;
               margin:6px 0; border:1px solid #e0e7ff;}
    .msg-bot {background:#f8fafc; border-radius:12px; padding:10px 14px;
              margin:6px 0; border:1px solid #eef0f3; white-space:pre-wrap;}
    footer {visibility:hidden;}
    """

    EXAMPLES = [
        "Which stocks have the highest predicted probability of a gain tomorrow?",
        "Which sector should I buy into for gains next week?",
        "Is now a good time to buy into the Energy sector?",
        "Why is XOM predicted the way it is?",
        "How confident are you about Microsoft?",
        "Should I sell or hold my Apple stock?",
    ]

    with gr.Blocks(css=CSS, theme=gr.themes.Soft(
            primary_hue="indigo", secondary_hue="slate")) as demo:
        gr.Markdown("# 🔍 stock-lens", elem_id="title-row")
        gr.Markdown("Ask in plain English — predictions, confidence, feature "
                    "explanations, sector rankings, or top stocks. "
                    "Data covers 2010–2016; “today” means the latest day in "
                    "the dataset.", elem_id="subtitle")

        chat = gr.Chatbot(label=None, height=440, show_copy_button=True,
                          bubble_full_width=False, type="messages")
        # session-scoped history of the user's questions
        history = gr.State([])

        with gr.Row():
            box = gr.Textbox(placeholder="e.g. Which sector looks best right "
                                         "now?  ·  Why is MSFT predicted up?",
                             scale=8, show_label=False, autofocus=True,
                             container=False)
            send = gr.Button("Send", variant="primary", scale=1, min_width=90)

        with gr.Row():
            clear = gr.Button("🗑 Clear chat", size="sm", scale=1)
            reask = gr.Button("↻ Re-ask last", size="sm", scale=1)

        gr.Markdown("**Recent questions this session** — click one to copy it "
                    "back into the box:")
        recents = gr.Radio(choices=[], label=None, interactive=True)

        gr.Examples(examples=EXAMPLES, inputs=box, label="Try one:")

        def respond(message, chat_msgs, hist):
            # answering a question and updating chat + session history
            message = (message or "").strip()
            if not message:
                return chat_msgs, hist, gr.update(), ""
            reply = answer_fn(message)
            chat_msgs = (chat_msgs or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply}]
            hist = ([message] + [h for h in hist if h != message])[:12]
            return chat_msgs, hist, gr.update(choices=hist, value=None), ""

        def do_clear():
            return [], [], gr.update(choices=[], value=None), ""

        def do_reask(chat_msgs, hist):
            # re-running the most recent question
            if not hist:
                return chat_msgs, hist, gr.update(), ""
            return respond(hist[0], chat_msgs, hist)

        def pick_recent(choice):
            # copying a past question back into the input box
            return choice or ""

        send.click(respond, [box, chat, history],
                   [chat, history, recents, box])
        box.submit(respond, [box, chat, history],
                   [chat, history, recents, box])
        clear.click(do_clear, None, [chat, history, recents, box])
        reask.click(do_reask, [chat, history], [chat, history, recents, box])
        recents.change(pick_recent, recents, box)

    return demo
