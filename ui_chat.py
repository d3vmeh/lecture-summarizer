import shelve


def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages",[])
    
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages