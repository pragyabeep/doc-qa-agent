"""Tools the agent can call. The model only *asks* for these; our code runs them."""
import ast
import json
import operator

from . import config as cfg
from .retrieval import get_index
from .tracing import observe

# ------------------------------------------------------------------ implementations
@observe(as_type="retriever", name="search_documents")
def search_documents(query: str, top_k: int | None = None) -> str:
    hits = get_index().search(query, k=top_k)
    if not hits:
        return "NO_RESULTS: no relevant passages found. Try different wording, or answer that the documents don't cover it."
    return "\n\n".join(f"[source: {h['source']} | section: {h['heading']}]\n{h['text']}" for h in hits)


@observe(as_type="tool", name="lookup_ticket")
def lookup_ticket(ticket_id: str) -> str:
    tickets = json.loads(cfg.TICKETS_FILE.read_text(encoding="utf-8"))
    key = ticket_id.strip().upper()
    if key not in tickets:
        raise ValueError(f"Ticket '{ticket_id}' not found. Ticket IDs look like T-1001.")
    return json.dumps({"ticket_id": key, **tickets[key]})


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed.")


@observe(as_type="tool", name="calculator")
def calculator(expression: str) -> str:
    """Safe arithmetic (no eval): parses the expression and only allows numbers and operators."""
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval").body)
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except SyntaxError:
        raise ValueError(f"Could not parse '{expression}'. Example of a valid input: 8 * 10 * 12")
    return str(round(result, 6)) if isinstance(result, float) else str(result)


REGISTRY = {"search_documents": search_documents, "lookup_ticket": lookup_ticket, "calculator": calculator}

# ------------------------------------------------------------------ what the model sees
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": ("Search the CloudSync product documentation (plans, pricing, billing, refunds, error codes, "
                        "security, SSO, API limits). Use this for ANY question about how CloudSync works. "
                        "Put exact identifiers such as error codes (E-4021) in the query verbatim."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, e.g. 'refund policy annual plan' or 'E-4021'."},
            "top_k": {"type": "integer", "description": "Number of passages to return (default 4)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_ticket",
        "description": ("Look up one support ticket by ID (format T-1001). Returns customer, subject, status, "
                        "priority, assignee and latest update. Use only when the user mentions a ticket ID."),
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string", "description": "Ticket ID such as T-1003."}}, "required": ["ticket_id"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": ("Evaluate an arithmetic expression exactly. ALWAYS use this for any calculation (totals, "
                        "price x users x months) instead of computing in your head. Example: '8 * 10 * 12'."),
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Numbers and + - * / ( ) only."}}, "required": ["expression"]}}},
]
