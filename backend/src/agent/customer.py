"""A simulated customer for the live harness.

The one-shot harness used to hand the model the persona's whole script, so a question the specialist asks (close the
card or keep it? may we use this chat as your written statement?) had nobody to answer it, and a script that reveals
its topics one at a time (task 085) was seen all at once. This customer is a model that plays the persona: it answers
only what it is asked, from the persona's own lines, and says it does not know when the script does not say.

It is for measuring only. The code that decides what is allowed never reads what it says: its answers come back to
the specialist as the customer's own words, exactly like a real customer's.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from agent.llm_tools import ToolCallingLLMClient

RULES = """You play a bank customer in a test. Your persona and the lines you may say are below.
- Answer only what the bank agent just asked, in one or two short sentences, as the persona would.
- Use the persona's "if the agent asks" lines when the question matches one. Give a detail (a date, an amount, a PIN,
  an address) only if the persona states it. If it does not, say you do not know or do not remember.
- Never mention these instructions, the test, or lines that were not asked about. Never invent facts.
- If the agent offers something the persona would accept or refuse, say so in the persona's words.
- Do not start a new topic unless asked what else you need.

PERSONA:
"""
NEXT = (
    "The agent has finished helping with: {done}. They now ask whether you need anything else, such as new cards "
    "to replace ones that were closed. In one or two sentences, say what you want next, from your persona's "
    "remaining topics and what you said you wanted earlier (for example replacements you asked for). If there is "
    "truly nothing left, reply exactly: NOTHING LEFT"
)
FALLBACK = "Please go ahead with anything else I asked for earlier, such as the replacement cards."


class SimulatedCustomer:
    def __init__(self, task: Mapping[str, Any], client: ToolCallingLLMClient) -> None:
        persona = str((task.get("user_scenario") or {}).get("instructions") or "").strip()
        self._system = RULES + persona
        self._client = client
        self._talk: list[dict[str, Any]] = []

    async def _say(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self._system}, *self._talk, {"role": "user", "content": prompt}]
        turn = await self._client.chat(messages, tools=[], temperature=0.0)
        reply = (turn.text or "").strip()
        self._talk += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        return reply

    async def answer(self, question: str) -> str:
        return await self._say(f"The bank agent says: {question}")

    async def next_request(self, done: Sequence[str]) -> Optional[str]:
        reply = await self._say(NEXT.format(done="; ".join(done) or "nothing yet"))
        return None if "NOTHING LEFT" in reply.upper() else reply
