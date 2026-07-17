#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10-item eval harness для Grounded Q&A бота.
Проганяє питання й перевіряє: (1) чи знайдено правильний документ,
(2) чи спрацював grounding на питанні без відповіді.

Запуск:  python3 eval.py
"""

from qa_bot import GroundedQABot

# (питання, очікуваний документ)  — None означає "відповіді немає" (grounding)
GOLDEN = [
    ("How do I reset my password?",           "Authentication.md"),
    ("How can I create an invoice?",           "Billing.md"),
    ("How do I delete a user?",                "Users.md"),
    ("How does API authentication work?",      "Authentication.md"),
    ("Which currencies are supported?",        "Payments.md"),
    ("How do I enable two-factor auth?",       "Authentication.md"),
    ("How can I issue a refund?",              "Billing.md"),
    ("How do I retry a failed payment?",       "Payments.md"),
    ("How do I configure webhooks?",           "Notifications.md"),
    ("What user roles exist?",                 "Users.md"),
    ("Do you have a mobile app?",              None),   # пастка
]


def main():
    bot = GroundedQABot().index()
    passed = 0
    print(f"{'#':>2}  {'ok':>3}  {'expected':<20} {'got':<20} question")
    print("-" * 88)
    for i, (q, expected) in enumerate(GOLDEN, 1):
        ans = bot.ask(q)
        got = ans.citations[0].split(" › ")[0] if ans.citations else "—"
        if expected is None:
            ok = not ans.grounded            # має бути "не знаю"
        else:
            ok = ans.grounded and got == expected
        passed += ok
        print(f"{i:>2}  {'✓' if ok else '✗':>3}  {str(expected):<20} {got:<20} {q}")
    print("-" * 88)
    print(f"Accuracy: {passed}/{len(GOLDEN)} = {passed / len(GOLDEN) * 100:.0f}%")


if __name__ == "__main__":
    main()
