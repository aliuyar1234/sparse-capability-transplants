from __future__ import annotations

import unittest

from src.data.build_alias_bank import alias_banks_hash, freeze_alias_banks


class AliasBankTests(unittest.TestCase):
    def test_alias_banks_are_disjoint_across_canonical_keys(self) -> None:
        alias_banks = freeze_alias_banks(
            {
                "tool_names": {
                    "send_email": ["dispatch_message", "transmit_mail", "compose_email"],
                    "set_alarm": ["schedule_alarm", "arm_clock", "wake_timer"],
                },
                "argument_names": {
                    "send_email.recipient": ["to", "target", "addressee"],
                    "send_email.subject": ["topic", "subject_line", "headline"],
                },
                "tool_descriptions": {
                    "send_email": [
                        "Send an email",
                        "Compose an electronic message",
                        "Dispatch mail",
                    ],
                },
            }
        )

        tool_train_aliases = alias_banks.banks["train"]["tool_names"]
        self.assertNotEqual(
            set(tool_train_aliases["send_email"]),
            set(tool_train_aliases["set_alarm"]),
        )
        self.assertEqual(len(alias_banks_hash(alias_banks)), 40)

    def test_alias_collisions_raise(self) -> None:
        with self.assertRaises(ValueError):
            freeze_alias_banks(
                {
                    "tool_names": {
                        "send_email": ["dispatch_message"],
                        "set_alarm": ["dispatch_message"],
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
