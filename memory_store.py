from collections import deque

# user_id → deque of messages
memory_store = {}
MAX_MEMORY = 7  # last 7 messages

def add_memory(user_id, role, message):
    if user_id not in memory_store:
        memory_store[user_id] = deque(maxlen=MAX_MEMORY)
    memory_store[user_id].append({"role": role, "msg": message})

def get_memory_text(user_id):
    if user_id not in memory_store:
        return ""
    return "\n".join([f"{m['role']}: {m['msg']}" for m in memory_store[user_id]])