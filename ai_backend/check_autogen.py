try:
    import autogen
    print("autogen (legacy) found")
except ImportError:
    print("autogen (legacy) NOT found")

try:
    import autogen_agentchat
    print("autogen_agentchat (v0.4) found")
except ImportError:
    print("autogen_agentchat (v0.4) NOT found")
