"""Assistant system prompt for tau2-bench."""

# Official tau2-bench multi-turn instruction (from tau2-bench/src/tau2/agent/llm_agent.py)
MULTI_TURN_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

def assistant_system_prompt(policy: str) -> str:
    return f"""
<instructions>
{MULTI_TURN_INSTRUCTION}
</instructions>

<policy>
{policy}
</policy>"""