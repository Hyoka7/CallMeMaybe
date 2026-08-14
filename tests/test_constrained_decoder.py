import unittest

from src.constrained_decoder import END, TrieNode


class TrieNodeTests(unittest.TestCase):
    def test_prefix_function_allows_child_and_end(self) -> None:
        root = TrieNode()
        root.insert([1, 2], "fn_add")
        root.insert([1, 2, 3], "fn_add_numbers")

        prefix = root.children[1].children[2]

        self.assertEqual(set(prefix.children), {END, 3})
        self.assertEqual(prefix.children[END].value, "fn_add")
        self.assertEqual(
            prefix.children[3].children[END].value,
            "fn_add_numbers",
        )


if __name__ == "__main__":
    unittest.main()
