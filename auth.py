#Authorization Module

import os

def signup(username, password):
    os.makedirs("data", exist_ok=True)
    path = "data/users.txt"
    if not os.path.exists(path):
        open(path, "w").close()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, p = line.strip().split(",")
            if u == username:
                return "Username exists."
    with open(path, "a") as f:
        f.write(f"{username},{password}\n")
    return "Account created."

def login(username, password):
    path = "data/users.txt"
    if not os.path.exists(path):
        return "No users."
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, p = line.strip().split(",")
            if u == username and p == password:
                return "Login successful."
    return "Invalid credentials."
