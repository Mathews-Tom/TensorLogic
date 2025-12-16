"""TensorLogic Knowledge Graph Query: Realistic Entity Lookup

This example shows how to query a knowledge graph for real-world questions:
- "Which employees work in Engineering?"
- "Who manages Bob?"
- "Which employees know Python AND work in Engineering?"

What you'll learn:
    1. How to encode entity relationships as matrices
    2. How to compose queries with AND/OR
    3. How to use EXISTS for "is there any X such that..."

The Key Insight:
    A knowledge graph is just a collection of relationship matrices.
    Queries are combinations of these matrices using logical operations.

Run this example:
    uv run python examples/03_knowledge_graph_query.py
"""

from __future__ import annotations

import numpy as np

from tensorlogic import create_backend, logical_and, logical_or
from tensorlogic.core.quantifiers import exists

# =============================================================================
# SETUP: A Small Company Knowledge Graph
# =============================================================================

backend = create_backend()
print(f"Backend: {type(backend).__name__}")
print()

# Entities (employees)
employees = ["Alice", "Bob", "Carol", "David", "Eve"]
N = len(employees)

# Departments
departments = ["Engineering", "Marketing", "Sales"]

# Skills
skills = ["Python", "SQL", "JavaScript", "Excel"]

# =============================================================================
# RELATIONS: Encode as Matrices
# =============================================================================

# works_in[person, dept] = 1.0 if person works in that department
#               Engineering  Marketing  Sales
works_in = np.array([
    [1., 0., 0.],  # Alice: Engineering
    [1., 0., 0.],  # Bob: Engineering
    [0., 1., 0.],  # Carol: Marketing
    [0., 0., 1.],  # David: Sales
    [1., 1., 0.],  # Eve: Engineering AND Marketing (cross-functional)
], dtype=np.float32)

# has_skill[person, skill] = 1.0 if person has that skill
#              Python  SQL  JavaScript  Excel
has_skill = np.array([
    [1., 1., 0., 0.],  # Alice: Python, SQL
    [1., 0., 1., 0.],  # Bob: Python, JavaScript
    [0., 0., 0., 1.],  # Carol: Excel
    [0., 1., 0., 1.],  # David: SQL, Excel
    [1., 1., 1., 0.],  # Eve: Python, SQL, JavaScript
], dtype=np.float32)

# manages[manager, report] = 1.0 if manager manages report
manages = np.array([
    [0., 1., 0., 0., 1.],  # Alice manages Bob, Eve
    [0., 0., 0., 0., 0.],  # Bob manages no one
    [0., 0., 0., 0., 0.],  # Carol manages no one
    [0., 0., 1., 0., 0.],  # David manages Carol
    [0., 0., 0., 0., 0.],  # Eve manages no one
], dtype=np.float32)

print("=== Company Knowledge Graph ===")
print()
print("Employees:", employees)
print("Departments:", departments)
print("Skills:", skills)
print()

# =============================================================================
# QUERY 1: "Which employees work in Engineering?"
# =============================================================================

print("=" * 60)
print("QUERY 1: Which employees work in Engineering?")
print("=" * 60)
print()

engineering_idx = departments.index("Engineering")
in_engineering = works_in[:, engineering_idx]

print("Result:")
for i, emp in enumerate(employees):
    if in_engineering[i] > 0:
        print(f"  - {emp}")
print()

# =============================================================================
# QUERY 2: "Who manages Bob?"
# =============================================================================

print("=" * 60)
print("QUERY 2: Who manages Bob?")
print("=" * 60)
print()

bob_idx = employees.index("Bob")
# managers[:, bob_idx] gives who manages Bob
bob_managers = manages[:, bob_idx]

print("Result:")
for i, emp in enumerate(employees):
    if bob_managers[i] > 0:
        print(f"  - {emp} manages Bob")
print()

# =============================================================================
# QUERY 3: "Which employees know Python AND work in Engineering?"
# =============================================================================

print("=" * 60)
print("QUERY 3: Which employees know Python AND work in Engineering?")
print("=" * 60)
print()

python_idx = skills.index("Python")
knows_python = has_skill[:, python_idx]

# Logical AND: must satisfy BOTH conditions
python_and_engineering = logical_and(knows_python, in_engineering, backend=backend)

print("Knows Python:", [employees[i] for i in range(N) if knows_python[i] > 0])
print("In Engineering:", [employees[i] for i in range(N) if in_engineering[i] > 0])
print()
print("Python AND Engineering:")
for i, emp in enumerate(employees):
    if np.asarray(python_and_engineering)[i] > 0:
        print(f"  - {emp}")
print()

# =============================================================================
# QUERY 4: "Which employees know Python OR SQL?"
# =============================================================================

print("=" * 60)
print("QUERY 4: Which employees know Python OR SQL?")
print("=" * 60)
print()

sql_idx = skills.index("SQL")
knows_sql = has_skill[:, sql_idx]

python_or_sql = logical_or(knows_python, knows_sql, backend=backend)

print("Python OR SQL:")
for i, emp in enumerate(employees):
    if np.asarray(python_or_sql)[i] > 0:
        print(f"  - {emp}")
print()

# =============================================================================
# QUERY 5: "Which departments have Python developers?" (EXISTS)
# =============================================================================

print("=" * 60)
print("QUERY 5: Which departments have Python developers?")
print("=" * 60)
print()

# For each department, check if ANY employee in that department knows Python
# dept_has_python[d] = EXISTS person: works_in[person, d] AND has_skill[person, Python]

# Broadcast: person dimension
# knows_python shape: (N_employees,)
# works_in shape: (N_employees, N_departments)
# We want: for each department, is there anyone who works there AND knows Python?

knows_python_expanded = knows_python.reshape(-1, 1)  # (N, 1)
python_in_dept = logical_and(knows_python_expanded, works_in, backend=backend)

# EXISTS over person dimension (axis 0)
dept_has_python = exists(np.asarray(python_in_dept), axis=0, backend=backend)

print("Result:")
for i, dept in enumerate(departments):
    status = "Yes" if np.asarray(dept_has_python)[i] > 0 else "No"
    print(f"  - {dept}: {status}")
print()

# =============================================================================
# QUERY 6: "Who is a second-level report?" (Transitive Query)
# =============================================================================

print("=" * 60)
print("QUERY 6: Who is managed by someone who is also managed? (2nd level)")
print("=" * 60)
print()

# manages_chain[x, z] = EXISTS y: manages[x, y] AND manages[y, z]
# This is matrix multiplication: manages @ manages
manages_2nd_level = backend.einsum('xy,yz->xz', manages, manages)
manages_2nd_level = backend.step(manages_2nd_level)  # Convert to boolean

print("Direct reports:")
for i, mgr in enumerate(employees):
    reports = [employees[j] for j in range(N) if manages[i, j] > 0]
    if reports:
        print(f"  - {mgr} manages: {', '.join(reports)}")

print()
print("Second-level reports (skip-level):")
for i, mgr in enumerate(employees):
    reports = [employees[j] for j in range(N) if np.asarray(manages_2nd_level)[i, j] > 0]
    if reports:
        print(f"  - {mgr} has skip-level: {', '.join(reports)}")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY: Knowledge Graph Query Patterns")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│  PATTERN                        TENSORLOGIC                 │
├─────────────────────────────────────────────────────────────┤
│  Simple lookup                  matrix[:, index]            │
│  A AND B                        logical_and(A, B)           │
│  A OR B                         logical_or(A, B)            │
│  EXISTS x: P(x)                 exists(P, axis)             │
│  Transitive (2-hop)             einsum('xy,yz->xz', R, R)   │
└─────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# Now that you understand KG queries:
#   - examples/04_recommendation_system.py (similarity-based recommendations)
#   - examples/knowledge_graph_reasoning.py (comprehensive 8-person example)
#   - docs/concepts/tensor-logic-mapping.md (mathematical foundations)
