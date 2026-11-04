"""Structured workflows: the model supplies facts, code makes the decisions.

A workflow is one kind of request written down as a recipe. The model's job shrinks to
what a model is good at: working out what the customer wants and filling in a typed form
of facts (which card, which transaction, what the customer said). The workflow's code does
the rest: look up the records, apply the bank's rules (fees, liability, who is eligible),
and build the exact list of changes. Nothing the model writes is a number the bank's rules
should decide.
"""
