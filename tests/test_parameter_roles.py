import unittest

from src.parameter_roles import build_regex_name_prompt
from src.parameter_roles import normalize_parameter_name


class ParameterRoleTests(unittest.TestCase):
    def test_normalizes_separators(self) -> None:
        self.assertEqual(normalize_parameter_name("reg_ex"), "regex")
        self.assertEqual(normalize_parameter_name("regex-pattern"), "regexpattern")

    def test_prompt_focuses_on_parameter_name(self) -> None:
        prompt = build_regex_name_prompt("reg_ex")

        self.assertIn("Parameter name: regex", prompt)
        self.assertIn("payload -> 1", prompt)
        self.assertIn("matchexpression -> 0", prompt)
        self.assertNotIn("replacement", prompt)


if __name__ == "__main__":
    unittest.main()
