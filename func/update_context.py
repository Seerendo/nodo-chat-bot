def update_context(context: str, user_input: str, bot_response: str, max_turns: int = 3):
    turns = context.strip().split("\n")
    turns.append(f"User: {user_input}")
    turns.append(f"Bot: {bot_response}")
    
    return "\n".join(turns[-2 * max_turns:])
